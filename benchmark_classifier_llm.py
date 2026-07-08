"""
benchmark_classifier_llm.py
----------------------------
Benchmark Small LLM + LoRA fine-tuning cho bài toán Level Classification.
Dùng PEFT (LoRA) để fine-tune các mô hình autoregressive nhỏ (0.5B–2B params)
trên tập Job Description. Tiết kiệm VRAM tối đa nhờ 4-bit quantization.

Sử dụng:
    python benchmark_classifier_llm.py [--config config_benchmark_llm.yaml]

Yêu cầu thư viện:
    pip install peft bitsandbytes accelerate

Phương pháp:
    - Load LLM (Qwen/Gemma) với 4-bit quantization (bitsandbytes)
    - Thêm LoRA adapter vào các lớp Q/V/K/O proj của attention
    - Thêm LinearClassificationHead lên embedding của token cuối cùng
    - Train chỉ param LoRA + head (~1-3% tổng params)
    - Evaluate bằng F1-weighted và Accuracy giống benchmark_classifier.py

Hàm chính:
    - build_lora_model()          : Load model + thêm LoRA + classifier head
    - LLMJobDataset               : Dataset cho LLM (dùng prompt template)
    - run_single_llm_benchmark()  : Train + evaluate 1 LLM
    - run_all_llm_benchmarks()    : Duyệt qua tất cả model trong config
    - print_results_table()       : In bảng so sánh
    - save_results()              : Lưu kết quả ra file
"""

import os
import sys
import json
import csv
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_dataset, set_seed,
    print_banner, check_device, format_time
)


# ----------------------------------------------------------------
# Dataset cho LLM (dùng text đơn giản, không dùng instruction template)
# ----------------------------------------------------------------
class LLMJobDataset:
    """
    Dataset cho LLM classification.

    Mỗi sample được tokenize trực tiếp từ văn bản JD (không dùng chat template).
    Label được mã hóa thành int và xử lý riêng trong classification head.

    Args:
        data                : List dict từ dataset
        tokenizer           : HuggingFace tokenizer của LLM
        level_labels        : Danh sách các nhãn level
        max_length          : Độ dài tối đa token
        truncation_strategy : "head", "tail", hoặc "head+tail"
    """

    def __init__(
        self,
        data: List[dict],
        tokenizer,
        level_labels: List[str],
        max_length: int = 512,
        truncation_strategy: str = "head+tail",
    ):
        self.tokenizer = tokenizer
        self.level_labels = [lv.upper() for lv in level_labels]
        self.max_length = max_length
        self.strategy = truncation_strategy
        self.samples = []
        skipped = 0

        for item in data:
            text = item.get("text", "").strip()
            level = str(item.get("level", "")).upper().strip()
            if not text or level not in self.level_labels:
                skipped += 1
                continue
            label_idx = self.level_labels.index(level)
            self.samples.append((text, label_idx))

        if skipped > 0:
            print(f"  [LLMDataset] Skipped {skipped} samples (empty/unknown level)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        text, label_idx = self.samples[idx]

        # Tokenize không dùng padding (sẽ pad trong collate_fn)
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]

        # Chiến lược cắt ghép head+tail
        max_tok = self.max_length - 1  # -1 cho EOS token
        if len(tokens) > max_tok:
            if self.strategy == "head":
                tokens = tokens[:max_tok]
            elif self.strategy == "tail":
                tokens = tokens[-max_tok:]
            else:  # head+tail
                head_size = min(128, max_tok // 4)
                tail_size = max_tok - head_size
                tokens = tokens[:head_size] + tokens[-tail_size:]

        # Thêm EOS token vào cuối (LLM dùng EOS thay vì SEP)
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 2
        tokens = tokens + [eos_id]

        return {
            "input_ids": tokens,
            "label": label_idx,
        }


def llm_collate_fn(batch, tokenizer):
    """
    Collate function cho LLM: left-padding (LLM convention) và tạo attention mask.

    LLM thường dùng left-padding thay vì right-padding của BERT
    để đảm bảo token cuối cùng luôn là token thực (dùng để lấy embedding phân loại).
    """
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    max_len = max(len(s["input_ids"]) for s in batch)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for s in batch:
        ids = s["input_ids"]
        pad_len = max_len - len(ids)
        # Left padding
        padded = [pad_id] * pad_len + ids
        mask = [0] * pad_len + [1] * len(ids)
        input_ids_list.append(padded)
        attention_mask_list.append(mask)
        labels_list.append(s["label"])

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }


# ----------------------------------------------------------------
# Build LLM với LoRA + Classifier Head
# ----------------------------------------------------------------
def build_lora_model(model_name: str, num_labels: int, model_cfg: dict, bench_cfg: dict, device: str):
    """
    Load LLM, thêm LoRA adapter và classification head.

    Returns:
        model      : LLM với LoRA adapter và classifier head đã gắn
        tokenizer  : Tokenizer tương ứng
        n_params   : Số param có thể train (LoRA + head)
    """
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    use_4bit = model_cfg.get("use_4bit", bench_cfg.get("use_4bit", True))

    print(f"  Load tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Đảm bảo pad token tồn tại
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = None
    if use_4bit:
        compute_dtype_str = bench_cfg.get("bnb_4bit_compute_dtype", "float16")
        compute_dtype = torch.float16 if compute_dtype_str == "float16" else torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"  4-bit quantization: ON ({compute_dtype_str})")
    else:
        print(f"  4-bit quantization: OFF")

    print(f"  Load LLM: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False  # Tắt cache khi train

    # LoRA config
    lora_r = model_cfg.get("lora_r", bench_cfg.get("lora_r", 16))
    lora_alpha = model_cfg.get("lora_alpha", bench_cfg.get("lora_alpha", 32))
    lora_dropout = model_cfg.get("lora_dropout", bench_cfg.get("lora_dropout", 0.05))
    lora_targets = model_cfg.get("lora_target_modules", bench_cfg.get("lora_target_modules", ["q_proj", "v_proj"]))

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_targets,
        task_type=TaskType.FEATURE_EXTRACTION,  # Extraction vì ta thêm head riêng
        bias="none",
    )
    model_with_lora = get_peft_model(base_model, lora_config)

    # Lấy hidden size từ model config
    hidden_size = base_model.config.hidden_size

    # Wrapper model thêm classification head
    class LLMClassifier(nn.Module):
        """
        Wrapper: LLM base + LoRA + Linear classifier head.
        Lấy embedding của token cuối cùng (EOS) để phân loại.
        """
        def __init__(self, base, hidden_size, num_labels):
            super().__init__()
            self.base = base
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_labels),
            )

        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Lấy hidden state của token cuối cùng (EOS hoặc last real token)
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]

            # Index của token thực tế cuối cùng trong mỗi sample
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_size = input_ids.shape[0]
            last_token_emb = last_hidden[
                torch.arange(batch_size, device=input_ids.device),
                seq_lengths,
            ]  # [B, hidden]

            logits = self.classifier(last_token_emb)  # [B, num_labels]
            return logits

    classifier = LLMClassifier(model_with_lora, hidden_size, num_labels)

    # Chỉ train param của LoRA adapter và classifier head
    for name, param in classifier.named_parameters():
        if "classifier" in name or "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in classifier.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/max(n_total,1):.2f}%)")

    return classifier, tokenizer, n_trainable


# ----------------------------------------------------------------
# Evaluate LLM model
# ----------------------------------------------------------------
def evaluate_llm(model, data_loader, device, level_labels, loss_fct=None):
    """Evaluate LLM classifier trên validation/test set."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_t.cpu().numpy())

            if loss_fct is not None:
                loss = loss_fct(logits, labels_t)
                total_loss += loss.item()
                n_batches += 1

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / max(n_batches, 1)

    return {"accuracy": acc, "f1_weighted": f1, "loss": avg_loss}


# ----------------------------------------------------------------
# Benchmark 1 LLM
# ----------------------------------------------------------------
def run_single_llm_benchmark(
    model_cfg: dict,
    bench_cfg: dict,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    level_labels: List[str],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Train và evaluate 1 LLM với LoRA.

    Args:
        model_cfg   : Config của model này
        bench_cfg   : Config chung của benchmark_llm
        train_data  : List dict train samples
        val_data    : List dict val samples
        test_data   : List dict test samples
        level_labels: Danh sách nhãn level
        device      : "cuda" hoặc "cpu"
        seed        : Random seed

    Returns:
        Dict kết quả: accuracy, f1_weighted, train_time, status, ...
    """
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    model_display_name = model_cfg.get("name", model_cfg["model_name"])
    model_name = model_cfg["model_name"]
    print_banner(f"LLM BENCHMARK: {model_display_name} ({model_name})")

    result = {
        "name": model_display_name,
        "model_name": model_name,
        "accuracy": 0.0,
        "f1_weighted": 0.0,
        "test_loss": 99.9,
        "train_time": "N/A",
        "train_time_sec": 0.0,
        "n_params": 0,
        "model_dir": "N/A",
        "status": "FAILED",
        "error": "",
    }

    try:
        set_seed(seed)

        num_epochs = bench_cfg.get("num_epochs", 3)
        max_length = bench_cfg.get("max_length", 512)
        truncation_strategy = bench_cfg.get("truncation_strategy", "head+tail")
        logging_steps = bench_cfg.get("logging_steps", 50)

        train_batch_size = model_cfg.get("train_batch_size", 4)
        eval_batch_size = model_cfg.get("eval_batch_size", 8)
        learning_rate = model_cfg.get("learning_rate", 2e-4)
        weight_decay = model_cfg.get("weight_decay", 0.01)
        warmup_ratio = model_cfg.get("warmup_ratio", 0.05)

        output_base = bench_cfg.get("output_dir", "./outputs/benchmark_llm")
        safe_name = model_display_name.replace("/", "_").replace(" ", "_").replace("-", "_")
        model_output_dir = os.path.join(output_base, safe_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Build model + tokenizer
        model, tokenizer, n_params = build_lora_model(
            model_name, len(level_labels), model_cfg, bench_cfg, device
        )
        model = model.to(device)

        # Dataset + DataLoader
        _collate = lambda b: llm_collate_fn(b, tokenizer)
        train_dataset = LLMJobDataset(train_data, tokenizer, level_labels, max_length, truncation_strategy)
        val_dataset   = LLMJobDataset(val_data,   tokenizer, level_labels, max_length, truncation_strategy)
        test_dataset  = LLMJobDataset(test_data,  tokenizer, level_labels, max_length, truncation_strategy)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,  collate_fn=_collate, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=eval_batch_size,  shuffle=False, collate_fn=_collate, num_workers=0)
        test_loader  = DataLoader(test_dataset,  batch_size=eval_batch_size,  shuffle=False, collate_fn=_collate, num_workers=0)

        # Optimizer + Scheduler (chỉ update param trainable)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate, weight_decay=weight_decay,
        )
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        loss_fct = torch.nn.CrossEntropyLoss()
        use_amp = device == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        best_f1 = -1.0
        best_metrics = {}
        best_model_dir = os.path.join(model_output_dir, "best_model")
        global_step = 0
        start_time = time.time()

        # Training loop
        print(f"[{model_display_name}] Bắt đầu train ({num_epochs} epochs)...")
        model.train()
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                global_step += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["labels"].to(device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = loss_fct(logits, labels_t)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fct(logits, labels_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

                if global_step % logging_steps == 0:
                    avg_l = epoch_loss / n_batches
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = format_time(time.time() - start_time)
                    print(f"  [E{epoch}/{num_epochs}] step={global_step} | loss={avg_l:.4f} | lr={lr_now:.2e} | {elapsed}")

            # Evaluate cuối epoch
            print(f"\n[{model_display_name}] Evaluate sau Epoch {epoch}...")
            metrics = evaluate_llm(model, val_loader, device, level_labels, loss_fct)
            print(f"  Loss={metrics['loss']:.4f} | Acc={metrics['accuracy']:.4f} | F1={metrics['f1_weighted']:.4f}")

            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                best_metrics = metrics.copy()
                # Lưu chỉ classifier head (LoRA đã nhúng vào base model)
                os.makedirs(best_model_dir, exist_ok=True)
                torch.save({
                    "classifier_state_dict": model.classifier.state_dict(),
                    "level_labels": level_labels,
                    "n_params": n_params,
                }, os.path.join(best_model_dir, "classifier_head.pt"))
                print(f"  *** Best! F1={best_f1:.4f} → Saved ***\n")

            model.train()

        train_time_sec = time.time() - start_time
        train_time_str = format_time(train_time_sec)
        print(f"[{model_display_name}] Hoàn thành sau {train_time_str}")

        # Evaluate trên tập Test
        print(f"\n[{model_display_name}] Đánh giá trên TẬP TEST độc lập...")
        test_metrics = evaluate_llm(model, test_loader, device, level_labels, loss_fct)
        print("=" * 60)
        print(f"  [{model_display_name}] KẾT QUẢ TRÊN TẬP TEST")
        print("=" * 60)
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print("=" * 60 + "\n")

        result.update({
            "accuracy": test_metrics.get("accuracy", 0.0),
            "f1_weighted": test_metrics.get("f1_weighted", 0.0),
            "test_loss": test_metrics.get("loss", 99.9),
            "train_time": train_time_str,
            "train_time_sec": train_time_sec,
            "n_params": n_params,
            "model_dir": best_model_dir,
            "status": "SUCCESS",
        })

        # Giải phóng GPU
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"\n[{model_display_name}] LỖI: {e}")
        print(error_msg)
        result["error"] = str(e)
        result["status"] = "FAILED"

    return result


# ----------------------------------------------------------------
# In bảng kết quả
# ----------------------------------------------------------------
def print_results_table(results: List[Dict[str, Any]]):
    """In bảng so sánh kết quả benchmark LLM."""
    print_banner("KẾT QUẢ BENCHMARK — LLM + LoRA")

    col_widths = {
        "name": 18, "model_name": 35, "accuracy": 12,
        "f1_weighted": 12, "test_loss": 12, "train_time": 12,
        "n_params": 14, "status": 8
    }

    def row_sep():
        return "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"

    def row_data(vals):
        cells = []
        for key, w in col_widths.items():
            v = str(vals.get(key, "N/A"))
            cells.append(f" {v[:w]:<{w}} ")
        return "|" + "|".join(cells) + "|"

    headers = {
        "name": "Model Name", "model_name": "HuggingFace ID",
        "accuracy": "Acc (Test)", "f1_weighted": "F1 (Test)",
        "test_loss": "Test Loss", "train_time": "Train Time",
        "n_params": "Trainable #", "status": "Status"
    }

    print(row_sep())
    print(row_data(headers))
    print(row_sep())

    sorted_results = sorted(
        results,
        key=lambda r: (r["status"] != "SUCCESS", -r.get("f1_weighted", 0))
    )
    for r in sorted_results:
        display = {
            "name": r["name"],
            "model_name": r["model_name"],
            "accuracy": f"{r.get('accuracy', 0):.4f}" if r["status"] == "SUCCESS" else "FAILED",
            "f1_weighted": f"{r.get('f1_weighted', 0):.4f}" if r["status"] == "SUCCESS" else "FAILED",
            "test_loss": f"{r.get('test_loss', 0):.4f}" if r["status"] == "SUCCESS" else "FAILED",
            "train_time": r.get("train_time", "N/A"),
            "n_params": f"{r.get('n_params', 0):,}" if r.get("n_params") else "N/A",
            "status": r["status"],
        }
        print(row_data(display))
    print(row_sep())

    success = [r for r in results if r["status"] == "SUCCESS"]
    if success:
        best = max(success, key=lambda r: r.get("f1_weighted", 0))
        fastest = min(success, key=lambda r: r.get("train_time_sec", float("inf")))
        print(f"\n🏆 Best F1    : {best['name']} ({best['f1_weighted']:.4f})")
        print(f"⚡ Fastest    : {fastest['name']} ({fastest['train_time']})")


# ----------------------------------------------------------------
# Lưu kết quả
# ----------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], output_dir: str):
    """Lưu kết quả benchmark ra CSV và TXT."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "benchmark_llm_results.csv")
    fieldnames = ["name", "model_name", "status", "accuracy", "f1_weighted",
                  "test_loss", "train_time", "train_time_sec", "n_params", "model_dir", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in ["accuracy", "f1_weighted", "test_loss"]:
                if isinstance(row.get(k), float):
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"\n[Benchmark LLM] CSV: {csv_path}")

    txt_path = os.path.join(output_dir, "benchmark_llm_results.txt")
    lines = ["=" * 100, "BENCHMARK LLM + LoRA RESULTS — Level Classifier", "=" * 100]
    sorted_r = sorted(results, key=lambda r: (r["status"] != "SUCCESS", -r.get("f1_weighted", 0)))
    for r in sorted_r:
        if r["status"] == "SUCCESS":
            lines.append(
                f"{r['name']:<20} {r['model_name']:<38} "
                f"Acc={r.get('accuracy', 0):.4f} F1={r.get('f1_weighted', 0):.4f} "
                f"Time={r.get('train_time', 'N/A')} Params={r.get('n_params', 0):,}"
            )
        else:
            lines.append(f"{r['name']:<20} {r['model_name']:<38} FAILED: {r.get('error', '')[:60]}")
    lines.append("=" * 100)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Benchmark LLM] TXT: {txt_path}")

    return csv_path, txt_path


# ----------------------------------------------------------------
# Chạy toàn bộ benchmark LLM
# ----------------------------------------------------------------
def run_all_llm_benchmarks(cfg: dict) -> List[Dict[str, Any]]:
    """
    Duyệt qua danh sách LLM trong config, train và evaluate từng cái.

    Args:
        cfg: Dict config đã load từ config_benchmark_llm.yaml

    Returns:
        List kết quả của từng model
    """
    print_banner("BẮT ĐẦU BENCHMARK LLM + LoRA — Level Classifier")

    bench_cfg = cfg.get("benchmark_llm", {})
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)

    set_seed(seed)
    device = check_device()

    level_labels = [lv.upper() for lv in bench_cfg.get("level_labels", [
        "INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"
    ])]

    # Load dataset
    print("[Benchmark LLM] Load dataset...")
    train_data, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.2),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
        level_labels=level_labels,
    )

    # Chia val → val + test (50/50)
    random.seed(seed)
    random.shuffle(val_data)
    split_idx = len(val_data) // 2
    test_data = val_data[:split_idx]
    val_data  = val_data[split_idx:]
    print(f"[Benchmark LLM] Split → Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    model_list = bench_cfg.get("models", [])
    enabled_models = [m for m in model_list if m.get("enabled", True)]
    print(f"\n[Benchmark LLM] Sẽ benchmark {len(enabled_models)}/{len(model_list)} model:")
    for m in enabled_models:
        print(f"  - {m.get('name', m['model_name'])} ({m['model_name']})")
    print()

    all_results = []
    overall_start = time.time()

    for i, model_cfg in enumerate(enabled_models, 1):
        model_name_display = model_cfg.get("name", model_cfg["model_name"])
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(enabled_models)}] {model_name_display}")
        print(f"{'='*60}")

        result = run_single_llm_benchmark(
            model_cfg=model_cfg,
            bench_cfg=bench_cfg,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            level_labels=level_labels,
            device=device,
            seed=seed,
        )
        all_results.append(result)

        if result["status"] == "SUCCESS":
            print(f"✓ {model_name_display}: Acc={result['accuracy']:.4f} | F1={result['f1_weighted']:.4f} | Time={result['train_time']}")
        else:
            print(f"✗ {model_name_display}: FAILED — {result.get('error', '')[:60]}")

    total_time = format_time(time.time() - overall_start)
    print(f"\n[Benchmark LLM] Tất cả model xong. Tổng thời gian: {total_time}")

    print_results_table(all_results)

    output_dir = bench_cfg.get("output_dir", "./outputs/benchmark_llm")
    save_results(all_results, output_dir)

    return all_results


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Benchmark Small LLM + LoRA cho Level Classification",
        epilog="""
Ví dụ:
  python benchmark_classifier_llm.py
  python benchmark_classifier_llm.py --config config_benchmark_llm.yaml
        """,
    )
    parser.add_argument("--config", type=str, default="config_benchmark_llm.yaml")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    if not os.path.exists(config_path):
        alt = str(Path(__file__).parent / args.config)
        if os.path.exists(alt):
            config_path = alt
        else:
            print(f"[LỖI] Không tìm thấy config: {args.config}")
            sys.exit(1)

    cfg = load_config(config_path)

    # Resolve dataset path
    dataset_path = cfg["data"]["dataset_path"]
    if not os.path.isabs(dataset_path):
        cfg["data"]["dataset_path"] = str(
            (Path(config_path).parent / dataset_path).resolve()
        )

    # Resolve output dir
    bench_out = cfg.get("benchmark_llm", {}).get("output_dir", "./outputs/benchmark_llm")
    if not os.path.isabs(bench_out):
        cfg["benchmark_llm"]["output_dir"] = str(
            (Path(config_path).parent / bench_out).resolve()
        )

    run_all_llm_benchmarks(cfg)
