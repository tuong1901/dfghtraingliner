"""
benchmark_classifier.py
------------------------
So sánh nhiều model architecture khác nhau trên bài toán Level Classification.
Mỗi model được train với cùng data split, sau đó so sánh qua bảng kết quả.

Sử dụng:
    python benchmark_classifier.py [--config config.yaml]

Các model được benchmark (cấu hình trong config.yaml → benchmark.models):
    - BERT-base          : bert-base-uncased
    - DistilBERT         : distilbert-base-uncased (nhanh ~2x, ~97% BERT)
    - RoBERTa-base       : roberta-base (thường tốt hơn BERT)
    - DeBERTa-v3-small   : microsoft/deberta-v3-small (SOTA nhỏ)
    - ELECTRA-small      : google/electra-small-discriminator (rất nhỏ, nhanh)
    - PhoBERT-base-v2    : vinai/phobert-base-v2 (Vietnamese, tắt mặc định)

Đầu vào  : text (Job Description dạng chuỗi thuần)
Đầu ra   : level ∈ {INTERN, FRESHER, JUNIOR, MIDDLE, SENIOR, LEAD, MANAGER, DIRECTOR, EXPERT, UNKNOWN}

Output:
    - Checkpoint mỗi model: ./outputs/benchmark/{model_name}/best_model/
    - Bảng kết quả tổng hợp: ./outputs/benchmark/benchmark_results.txt + .csv

Hàm chính:
    - run_single_benchmark() : Train và evaluate 1 model
    - run_all_benchmarks()   : Duyệt qua tất cả model trong config, gọi từng cái
    - print_results_table()  : In bảng so sánh đẹp ra terminal
    - save_results()         : Lưu kết quả ra file txt và csv
"""

import os
import sys
import json
import csv
import time
import copy
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_dataset, set_seed,
    print_banner, check_device, format_time
)
# Tái dùng Dataset và các hàm từ train_classifier.py
from train_classifier import JobLevelDataset, collate_fn, evaluate, OrdinalLoss


# ----------------------------------------------------------------
# Train + Evaluate 1 model trong benchmark
# ----------------------------------------------------------------
def run_single_benchmark(
    model_cfg: dict,
    benchmark_cfg: dict,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    level_labels: List[str],
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Train một model architecture trên tập train, evaluate trên tập val.
    Trả về dict kết quả để tổng hợp vào bảng benchmark.

    Args:
        model_cfg      : Config của model này (name, model_name, batch_size, lr...)
        benchmark_cfg  : Config chung của benchmark (epochs, max_length, output_dir...)
        train_data     : List dict train samples (đã split từ trước)
        val_data       : List dict val samples
        level_labels   : Danh sách các level
        device         : "cuda" hoặc "cpu"
        seed           : Random seed

    Returns:
        Dict kết quả: {
            "name": str,
            "model_name": str,
            "accuracy": float,
            "f1_weighted": float,
            "val_loss": float,
            "train_time": str,
            "train_time_sec": float,
            "model_dir": str,
            "status": "SUCCESS" | "FAILED",
            "error": str (nếu FAILED),
        }
    """
    model_display_name = model_cfg.get("name", model_cfg["model_name"])
    model_name = model_cfg["model_name"]

    print_banner(f"BENCHMARK: {model_display_name} ({model_name})")

    result = {
        "name": model_display_name,
        "model_name": model_name,
        "accuracy": 0.0,
        "f1_weighted": 0.0,
        "test_loss": 99.9,
        "train_time": "N/A",
        "train_time_sec": 0.0,
        "model_dir": "N/A",
        "status": "FAILED",
        "error": "",
    }

    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        set_seed(seed)

        # --- Hyperparameters (model override > benchmark default) ---
        num_epochs = benchmark_cfg.get("num_epochs", 3)
        max_length = benchmark_cfg.get("max_length", 512)
        truncation_strategy = benchmark_cfg.get("truncation_strategy", "head+tail")
        logging_steps = benchmark_cfg.get("logging_steps", 100)
        lambda_penalty = benchmark_cfg.get("lambda_penalty", 1.0)
        print(f"[{model_display_name}] Ordinal lambda_penalty: {lambda_penalty}")

        train_batch_size = model_cfg.get("train_batch_size", 16)
        eval_batch_size = model_cfg.get("eval_batch_size", 32)
        learning_rate = model_cfg.get("learning_rate", 2e-5)
        weight_decay = model_cfg.get("weight_decay", 0.01)
        warmup_ratio = model_cfg.get("warmup_ratio", 0.1)

        output_base = benchmark_cfg.get("output_dir", "./outputs/benchmark")
        safe_name = model_display_name.replace("/", "_").replace(" ", "_").replace("-", "_")
        model_output_dir = os.path.join(output_base, safe_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # --- Tokenizer ---
        print(f"[{model_display_name}] Load tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- Dataset + DataLoader ---
        print(f"[{model_display_name}] Tạo dataset...")
        train_dataset = JobLevelDataset(
            train_data, tokenizer, level_labels, max_length, truncation_strategy
        )
        val_dataset = JobLevelDataset(
            val_data, tokenizer, level_labels, max_length, truncation_strategy
        )

        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=eval_batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0,
        )

        # --- Model ---
        num_labels = len(level_labels)
        id2label = {i: lv for i, lv in enumerate(level_labels)}
        label2id = {lv: i for i, lv in enumerate(level_labels)}

        print(f"[{model_display_name}] Load model ({num_labels} classes)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model = model.to(device)

        # Tính class weights để xử lý mất cân bằng nhãn
        print(f"[{model_display_name}] Tính class weights xử lý mất cân bằng...")
        from sklearn.utils.class_weight import compute_class_weight
        from torch import nn
        train_labels = [s[1] for s in train_dataset.samples]
        
        unique_train_labels = np.unique(train_labels)
        computed_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_train_labels,
            y=train_labels
        )
        
        full_weights = np.ones(num_labels, dtype=np.float32)
        for cls_id, w in zip(unique_train_labels, computed_weights):
            full_weights[cls_id] = w
            
        class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)
        weights_dict = {level_labels[i]: round(class_weights[i].item(), 4) for i in range(num_labels)}
        print(f"  Class weights: {weights_dict}")

        # Đếm số params
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{model_display_name}] Trainable params: {n_params:,}")

        # --- Optimizer & Scheduler ---
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Mixed precision
        use_amp = device == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # --- Training Loop ---
        print(f"[{model_display_name}] Bắt đầu train ({num_epochs} epochs)...")
        best_f1 = 0.0
        best_metrics = {}
        best_model_dir = os.path.join(model_output_dir, "best_model")
        global_step = 0
        start_time = time.time()

        # Custom loss
        loss_fct = OrdinalLoss(class_weights=class_weights, level_labels=level_labels, lambda_penalty=lambda_penalty).to(device)

        model.train()
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                global_step += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["labels"].to(device)

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels_t,
                }
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(**kwargs)
                        loss = loss_fct(outputs.logits, labels_t)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(**kwargs)
                    loss = loss_fct(outputs.logits, labels_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1

                if global_step % logging_steps == 0:
                    avg_l = epoch_loss / n_batches
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = format_time(time.time() - start_time)
                    print(
                        f"  [E{epoch}/{num_epochs}] step={global_step} | "
                        f"loss={avg_l:.4f} | lr={lr_now:.2e} | {elapsed}"
                    )

            # Evaluate cuối epoch
            print(f"\n[{model_display_name}] Evaluate sau Epoch {epoch}...")
            metrics = evaluate(model, val_loader, device, level_labels, loss_fct=loss_fct)
            print(
                f"  Loss={metrics['loss']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1_weighted']:.4f}"
            )

            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                best_metrics = metrics.copy()
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print(f"  *** Best! F1={best_f1:.4f} -> Saved ***\n")

            model.train()

        train_time_sec = time.time() - start_time
        train_time_str = format_time(train_time_sec)
        print(f"[{model_display_name}] Hoàn thành sau {train_time_str}")

        # Lưu label map
        label_map_path = os.path.join(best_model_dir, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({
                "level_labels": level_labels,
                "id2label": id2label,
                "label2id": label2id,
                "n_params": n_params,
            }, f, ensure_ascii=False, indent=2)

        # Đánh giá khách quan trên tập test bằng best model vừa lưu
        print(f"[{model_display_name}] Đánh giá khách quan trên tập TEST thực tế bằng best model...")
        best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
        best_tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        
        test_dataset = JobLevelDataset(
            test_data, best_tokenizer, level_labels, max_length, truncation_strategy
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        test_metrics = evaluate(best_model, test_loader, device, level_labels, loss_fct=loss_fct)
        print("\n" + "="*60)
        print(f"  [{model_display_name}] KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (TEST SET)")
        print("="*60)
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print("="*60 + "\n")

        result.update({
            "accuracy": test_metrics.get("accuracy", 0.0),
            "f1_weighted": test_metrics.get("f1_weighted", 0.0),
            "test_loss": test_metrics.get("loss", 99.9),
            "train_time": train_time_str,
            "train_time_sec": train_time_sec,
            "model_dir": best_model_dir,
            "n_params": n_params,
            "status": "SUCCESS",
        })

        # Giải phóng GPU memory
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
    """
    In bảng so sánh kết quả benchmark dạng ASCII table đẹp.

    Args:
        results: List các dict kết quả từ run_single_benchmark()
    """
    print_banner("KẾT QUẢ BENCHMARK")

    # Header
    col_widths = {
        "name": 20, "model_name": 32, "accuracy": 12,
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
        "n_params": "# Params", "status": "Status"
    }

    print(row_sep())
    print(row_data(headers))
    print(row_sep())

    # Sắp xếp theo F1 giảm dần (SUCCESS trước, FAILED sau)
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

    # Tìm winner
    success_results = [r for r in results if r["status"] == "SUCCESS"]
    if success_results:
        best = max(success_results, key=lambda r: r.get("f1_weighted", 0))
        fastest = min(success_results, key=lambda r: r.get("train_time_sec", float("inf")))
        print(f"\n🏆 Best F1    : {best['name']} ({best['f1_weighted']:.4f})")
        print(f"⚡ Fastest    : {fastest['name']} ({fastest['train_time']})")


# ----------------------------------------------------------------
# Lưu kết quả ra file
# ----------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Lưu kết quả benchmark ra 2 file:
    - benchmark_results.txt : Bảng ASCII (dễ đọc)
    - benchmark_results.csv : CSV (dễ import vào Excel/Sheets)

    Args:
        results   : List kết quả benchmark
        output_dir: Thư mục output
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = [
        "name", "model_name", "status",
        "accuracy", "f1_weighted", "test_loss",
        "train_time", "train_time_sec", "n_params",
        "model_dir", "error"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            # Format float đẹp
            row = dict(r)
            for k in ["accuracy", "f1_weighted", "test_loss"]:
                if isinstance(row.get(k), float):
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)

    print(f"\n[Benchmark] Kết quả CSV: {csv_path}")

    # --- TXT (bảng ASCII) ---
    txt_path = os.path.join(output_dir, "benchmark_results.txt")
    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK RESULTS - Level Classifier")
    lines.append("=" * 100)
    lines.append(f"{'Model':<22} {'HuggingFace ID':<35} {'Acc (Test)':>10} {'F1 (Test)':>10} {'Loss (Test)':>12} {'Time':>12} {'Params':>14} {'Status'}")
    lines.append("-" * 100)

    sorted_r = sorted(results, key=lambda r: (r["status"] != "SUCCESS", -r.get("f1_weighted", 0)))
    for r in sorted_r:
        if r["status"] == "SUCCESS":
            lines.append(
                f"{r['name']:<22} {r['model_name']:<35} "
                f"{r.get('accuracy', 0):>10.4f} {r.get('f1_weighted', 0):>10.4f} "
                f"{r.get('test_loss', 0):>12.4f} {r.get('train_time', 'N/A'):>12} "
                f"{r.get('n_params', 0):>14,} {'OK'}"
            )
        else:
            lines.append(
                f"{r['name']:<22} {r['model_name']:<35} "
                f"{'FAILED':>8} {'FAILED':>8} {'FAILED':>8} {'N/A':>12} "
                f"{'N/A':>14} FAILED"
            )
            if r.get("error"):
                lines.append(f"  Error: {r['error'][:80]}")

    lines.append("=" * 100)

    success_results = [r for r in results if r["status"] == "SUCCESS"]
    if success_results:
        best = max(success_results, key=lambda r: r.get("f1_weighted", 0))
        fastest = min(success_results, key=lambda r: r.get("train_time_sec", float("inf")))
        lines.append(f"Best F1     : {best['name']} - F1={best['f1_weighted']:.4f}")
        lines.append(f"Fastest     : {fastest['name']} - {fastest['train_time']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Benchmark] Kết quả TXT: {txt_path}")

    return csv_path, txt_path


# ----------------------------------------------------------------
# Chạy toàn bộ benchmark
# ----------------------------------------------------------------
def run_all_benchmarks(cfg: dict) -> List[Dict[str, Any]]:
    """
    Duyệt qua danh sách model trong config, train và evaluate từng cái,
    in bảng kết quả và lưu file.

    Args:
        cfg: Dict config đã load từ config.yaml

    Returns:
        List kết quả của từng model
    """
    print_banner("BẮT ĐẦU BENCHMARK LEVEL CLASSIFIER")

    benchmark_cfg = cfg.get("benchmark", {})
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)

    set_seed(seed)
    device = check_device()

    # Level labels từ benchmark config
    level_labels = [
        lv.upper() for lv in benchmark_cfg.get(
            "level_labels",
            cfg.get("classifier", {}).get("level_labels", [
                "INTERN", "FRESHER", "JUNIOR", "MIDDLE",
                "SENIOR", "LEAD", "MANAGER", "DIRECTOR", "EXPERT", "UNKNOWN"
            ])
        )
    ]

    # Load dataset 1 lần, dùng chung cho tất cả model
    print("[Benchmark] Load dataset...")
    train_data, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.2), # default to 0.2 if not specified to allow 10/10 split
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
        level_labels=level_labels,
    )

    # Chia val_data thành val_data và test_data (50% validation, 50% test)
    # Giúp đánh giá khách quan trên tập test chưa từng dùng để chọn best model.
    random.seed(seed)
    random.shuffle(val_data)
    split_idx = len(val_data) // 2
    test_data = val_data[:split_idx]
    val_data = val_data[split_idx:]
    print(f"[Benchmark] Thực tế split -> Val: {len(val_data)} | Test: {len(test_data)}")

    # Danh sách model từ config
    model_list = benchmark_cfg.get("models", [])
    if not model_list:
        print("[LỖI] Không có model nào trong benchmark.models! Kiểm tra config.yaml")
        return []

    # Chỉ lấy model có enabled: true
    enabled_models = [m for m in model_list if m.get("enabled", True)]
    print(f"\n[Benchmark] Sẽ benchmark {len(enabled_models)}/{len(model_list)} model:")
    for m in enabled_models:
        print(f"  - {m.get('name', m['model_name'])} ({m['model_name']})")
    print()

    # Chạy từng model
    all_results = []
    overall_start = time.time()

    for i, model_cfg in enumerate(enabled_models, 1):
        model_name_display = model_cfg.get("name", model_cfg["model_name"])
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(enabled_models)}] {model_name_display}")
        print(f"{'='*60}")

        result = run_single_benchmark(
            model_cfg=model_cfg,
            benchmark_cfg=benchmark_cfg,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            level_labels=level_labels,
            device=device,
            seed=seed,
        )
        all_results.append(result)

        # In kết quả nhanh sau mỗi model
        if result["status"] == "SUCCESS":
            print(
                f"✓ {model_name_display}: "
                f"Acc={result['accuracy']:.4f} | "
                f"F1={result['f1_weighted']:.4f} | "
                f"Time={result['train_time']}"
            )
        else:
            print(f"✗ {model_name_display}: FAILED - {result.get('error', '')[:60]}")

    total_time = format_time(time.time() - overall_start)
    print(f"\n[Benchmark] Tất cả model đã xong. Tổng thời gian: {total_time}")

    # In bảng kết quả
    print_results_table(all_results)

    # Lưu kết quả
    output_dir = benchmark_cfg.get("output_dir", "./outputs/benchmark")
    save_results(all_results, output_dir)

    return all_results


# ----------------------------------------------------------------
# Entry point độc lập
# ----------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description="Benchmark nhiều Level Classifier model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python benchmark_classifier.py                         # Dùng config.yaml
  python benchmark_classifier.py --config my_cfg.yaml   # Config khác
        """
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Đường dẫn tới file config.yaml"
    )
    args = parser.parse_args()

    # Resolve config path
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
    bench_out = cfg.get("benchmark", {}).get("output_dir", "./outputs/benchmark")
    if not os.path.isabs(bench_out):
        cfg["benchmark"]["output_dir"] = str(
            (Path(config_path).parent / bench_out).resolve()
        )

    run_all_benchmarks(cfg)
