"""
train_classifier.py
-------------------
Script train Level Classifier (BERT-based) cho bài toán phân loại
cấp bậc công việc (INTERN / FRESHER / JUNIOR / MIDDLE / SENIOR / ...).

Sử dụng:
    python train_classifier.py [--config config.yaml]

Phương pháp:
    - Dùng một BERT/RoBERTa model làm encoder
    - Thêm Linear classification head lên [CLS] token
    - Finetune end-to-end với CrossEntropyLoss

Chiến lược truncation:
    - "head"     : Lấy max_length token từ đầu text
    - "tail"     : Lấy max_length token từ cuối text
    - "head+tail": Lấy nửa đầu + nửa cuối (tốt nhất cho JD dài)

Dataset format đầu vào:
    {
        "text": "...",
        "level": "SENIOR"    # Nhãn phân loại
    }

Output:
    Model lưu tại output_dir (mặc định: ./outputs/classifier)

Hàm chính:
    - JobLevelDataset       : Dataset class
    - train_classifier()    : Training loop chính
    - evaluate()            : Tính metrics trên val set
    - quick_test_classifier(): Test nhanh sau khi train
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np

# Thêm thư mục cha vào sys.path để import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_dataset, set_seed, normalize_level,
    print_banner, check_device, format_time
)


# ----------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------
class JobLevelDataset:
    """
    PyTorch Dataset cho Level Classification.

    Mỗi sample là một cặp (text, label_index).
    Hỗ trợ 3 chiến lược truncation: head, tail, head+tail.

    Args:
        data            : List dict từ dataset
        tokenizer       : HuggingFace tokenizer
        level_labels    : List các level theo thứ tự (để map sang index)
        max_length      : Độ dài tối đa token
        truncation_strategy: "head", "tail", hoặc "head+tail"
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
            print(f"[Classifier] Bỏ qua {skipped} sample không hợp lệ")
        
        # Thống kê phân bố nhãn
        label_counts = Counter(s[1] for s in self.samples)
        print("[Classifier] Phân bố nhãn trong dataset này:")
        for idx, count in sorted(label_counts.items()):
            print(f"    {self.level_labels[idx]}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, label = self.samples[idx]
        encoding = self._tokenize(text)
        encoding["labels"] = label
        return encoding

    def _tokenize(self, text: str) -> Dict[str, Any]:
        """
        Tokenize text với chiến lược truncation phù hợp.
        
        Với "head+tail": lấy 128 token đầu + phần còn lại từ cuối để đủ max_length.
        Điều này giúp model thấy cả phần mô tả công việc lẫn yêu cầu.
        """
        if self.strategy == "head":
            return self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        
        elif self.strategy == "tail":
            # Tokenize không truncate trước, rồi lấy tail
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # Giữ [CLS] ở đầu, max_length-2 token cuối, [SEP] ở cuối
            max_tokens = self.max_length - 2
            if len(tokens) > max_tokens:
                tokens = tokens[-max_tokens:]
            
            # Tạo lại encoding
            tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(tokens)
            
            # Padding
            pad_len = self.max_length - len(tokens)
            tokens += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            
            import torch
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
            }
        
        else:  # "head+tail" (mặc định)
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            max_tokens = self.max_length - 2  # trừ [CLS] và [SEP]
            
            if len(tokens) <= max_tokens:
                # Không cần truncate
                pass
            else:
                # Lấy 128 đầu + phần cuối
                head_size = min(128, max_tokens // 2)
                tail_size = max_tokens - head_size
                tokens = tokens[:head_size] + tokens[-tail_size:]
            
            tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(tokens)
            
            # Padding
            pad_len = self.max_length - len(tokens)
            tokens += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            
            import torch
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
            }


# ----------------------------------------------------------------
# Collate function cho DataLoader
# ----------------------------------------------------------------
def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Gộp các sample thành batch tensor.
    
    Args:
        batch: List các dict từ JobLevelDataset.__getitem__
    
    Returns:
        Dict batch với các tensor đã gộp
    """
    import torch
    
    input_ids = torch.cat([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    # Token type ids nếu có (BERT cần, RoBERTa không cần)
    if "token_type_ids" in batch[0]:
        result["token_type_ids"] = torch.cat([b["token_type_ids"] for b in batch], dim=0)
    
    return result


# ----------------------------------------------------------------
# Evaluate function
# ----------------------------------------------------------------
def evaluate(
    model,
    dataloader,
    device: str,
    level_labels: List[str],
) -> Dict[str, float]:
    """
    Đánh giá model trên validation set.
    Tính accuracy, weighted F1, và per-class F1.

    Args:
        model       : BERT classification model
        dataloader  : Val DataLoader
        device      : "cuda" hoặc "cpu"
        level_labels: Danh sách các level

    Returns:
        Dict metrics: {"accuracy": float, "f1_weighted": float, "loss": float}
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import classification_report, f1_score
    
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            
            outputs = model(**kwargs)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    # Per-class report
    present_labels = sorted(set(all_labels))
    present_names = [level_labels[i] for i in present_labels]
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    print("\n[Classifier] Classification Report:")
    print(report)
    
    model.train()
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_weighted": f1,
    }


# ----------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------
def train_classifier(cfg: dict):
    """
    Train BERT-based Level Classifier.

    Pipeline:
    1. Load & chia dataset
    2. Tokenize + tạo DataLoader
    3. Load pretrained BERT với classification head
    4. Training loop với early stopping
    5. Lưu best model

    Args:
        cfg: Dict config đã load từ config.yaml
    """
    print_banner("TRAINING LEVEL CLASSIFIER")
    
    # --- Import PyTorch + Transformers ---
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as e:
        print(f"\n[LỖI] Thiếu thư viện: {e}")
        print("Chạy: pip install torch transformers scikit-learn")
        sys.exit(1)
    
    ccfg = cfg["classifier"]
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)
    
    set_seed(seed)
    device = check_device()
    
    # 1. Level labels
    level_labels = [lv.upper() for lv in ccfg.get("level_labels", [
        "INTERN", "FRESHER", "JUNIOR", "MIDDLE",
        "SENIOR", "LEAD", "MANAGER", "DIRECTOR", "EXPERT", "UNKNOWN"
    ])]
    num_labels = len(level_labels)
    print(f"[Classifier] Số lớp phân loại: {num_labels}")
    print(f"[Classifier] Các level: {level_labels}")
    
    # 2. Load dataset (chỉ giữ sample có level hợp lệ)
    train_data, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.1),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
        level_labels=level_labels,
    )
    
    # 3. Tokenizer
    model_name = ccfg.get("model_name", "bert-base-uncased")
    max_length = ccfg.get("max_length", 512)
    truncation_strategy = ccfg.get("truncation_strategy", "head+tail")
    
    print(f"\n[Classifier] Load tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 4. Dataset và DataLoader
    train_batch_size = ccfg.get("train_batch_size", 16)
    eval_batch_size = ccfg.get("eval_batch_size", 32)
    
    print("\n[Classifier] Chuẩn bị train dataset...")
    train_dataset = JobLevelDataset(
        train_data, tokenizer, level_labels, max_length, truncation_strategy
    )
    print("\n[Classifier] Chuẩn bị val dataset...")
    val_dataset = JobLevelDataset(
        val_data, tokenizer, level_labels, max_length, truncation_strategy
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # 5. Model
    print(f"\n[Classifier] Load model: {model_name}")
    
    # Map level labels thành id2label và label2id cho model config
    id2label = {i: lv for i, lv in enumerate(level_labels)}
    label2id = {lv: i for i, lv in enumerate(level_labels)}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    
    # 6. Optimizer & Scheduler
    num_epochs = ccfg.get("num_epochs", 5)
    learning_rate = ccfg.get("learning_rate", 2e-5)
    weight_decay = ccfg.get("weight_decay", 0.01)
    warmup_ratio = ccfg.get("warmup_ratio", 0.1)
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # 7. Setup output và logging
    output_dir = ccfg.get("output_dir", "./outputs/classifier")
    os.makedirs(output_dir, exist_ok=True)
    eval_steps = ccfg.get("eval_steps", 200)
    logging_steps = ccfg.get("logging_steps", 50)
    save_steps = ccfg.get("save_steps", 500)
    
    # Wandb
    if ccfg.get("wandb_project"):
        try:
            import wandb
            wandb.init(project=ccfg["wandb_project"], config=ccfg)
        except ImportError:
            print("[CẢNH BÁO] wandb chưa cài, bỏ qua")
    
    # 8. Training loop
    print(f"\n[Classifier] Bắt đầu training...")
    print(f"  Total steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"  Eval mỗi {eval_steps} steps | Log mỗi {logging_steps} steps\n")
    
    best_f1 = 0.0
    best_model_dir = os.path.join(output_dir, "best_model")
    global_step = 0
    start_time = time.time()
    
    # Mixed precision
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            global_step += 1
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**kwargs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**kwargs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Logging
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / n_batches
                elapsed = format_time(time.time() - start_time)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [Epoch {epoch}/{num_epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Time: {elapsed}"
                )
            
            # Evaluation
            if global_step % eval_steps == 0:
                print(f"\n[Classifier] Đánh giá tại step {global_step}...")
                metrics = evaluate(model, val_loader, device, level_labels)
                print(
                    f"  Val Loss: {metrics['loss']:.4f} | "
                    f"Accuracy: {metrics['accuracy']:.4f} | "
                    f"F1(weighted): {metrics['f1_weighted']:.4f}"
                )
                
                # Lưu best model
                if metrics["f1_weighted"] > best_f1:
                    best_f1 = metrics["f1_weighted"]
                    os.makedirs(best_model_dir, exist_ok=True)
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                    print(f"  *** Best model mới! F1={best_f1:.4f} -> Lưu tại {best_model_dir} ***\n")
                
                model.train()
        
        # Eval cuối mỗi epoch
        print(f"\n[Classifier] Cuối Epoch {epoch} - Đánh giá...")
        metrics = evaluate(model, val_loader, device, level_labels)
        print(
            f"  Val Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1(weighted): {metrics['f1_weighted']:.4f}"
        )
        
        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"  *** Best model mới! F1={best_f1:.4f} -> Lưu tại {best_model_dir} ***\n")
    
    elapsed = format_time(time.time() - start_time)
    print(f"\n[Classifier] Training hoàn tất sau {elapsed}")
    print(f"[Classifier] Best F1 (weighted): {best_f1:.4f}")
    print(f"[Classifier] Best model: {best_model_dir}")
    
    # Lưu label mapping vào file json riêng
    label_map_path = os.path.join(best_model_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "level_labels": level_labels,
            "id2label": id2label,
            "label2id": label2id,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Classifier] Label map: {label_map_path}")
    
    return best_model_dir


# ----------------------------------------------------------------
# Quick inference test
# ----------------------------------------------------------------
def quick_test_classifier(model_dir: str, level_labels: List[str]):
    """
    Test nhanh model Classifier với một số câu mẫu.
    
    Args:
        model_dir   : Thư mục chứa model đã train
        level_labels: Danh sách các level
    """
    print_banner("QUICK TEST LEVEL CLASSIFIER")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        
        test_cases = [
            "Looking for a Senior Python developer with 5+ years of experience",
            "Junior Frontend Developer - Entry level position, 0-1 year experience",
            "Engineering Manager to lead a team of 10+ engineers",
            "Intern position - fresh graduates welcome, no experience required",
        ]
        
        level_labels_upper = [lv.upper() for lv in level_labels]
        
        for text in test_cases:
            inputs = tokenizer(
                text, return_tensors="pt",
                max_length=512, truncation=True, padding="max_length"
            )
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            pred_label = level_labels_upper[pred_idx]
            confidence = probs[pred_idx].item()
            
            print(f"  Input : {text[:80]}...")
            print(f"  Predict: {pred_label} (confidence: {confidence:.3f})\n")
    
    except Exception as e:
        print(f"[CẢNH BÁO] Quick test thất bại: {e}")


# ----------------------------------------------------------------
# Entry point độc lập
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Level Classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Đường dẫn config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    best_dir = train_classifier(cfg)
    
    level_labels = cfg["classifier"].get("level_labels", [
        "INTERN", "FRESHER", "JUNIOR", "MIDDLE",
        "SENIOR", "LEAD", "MANAGER", "DIRECTOR", "EXPERT", "UNKNOWN"
    ])
    quick_test_classifier(best_dir, level_labels)
