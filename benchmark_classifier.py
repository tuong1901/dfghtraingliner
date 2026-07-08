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
from train_classifier import JobLevelDataset, collate_fn, OrdinalLoss

# ----------------------------------------------------------------
# Hàm Evaluate cục bộ tích hợp đầy đủ metrics & Confusion Matrix
# ----------------------------------------------------------------
def evaluate(
    model,
    dataloader,
    device: str,
    level_labels: List[str],
    loss_fct=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Đánh giá model trên validation/test set.
    Trả về đầy đủ metrics bao gồm per-class, confusion matrix,
    MCC, Kappa, Macro F1, Weighted F1, Top-2 Accuracy.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (
        classification_report, f1_score, accuracy_score,
        confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
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
            logits = outputs.logits

            if loss_fct is not None:
                loss = loss_fct(logits, labels)
            else:
                loss = outputs.loss

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
            n_batches += 1

    # --- Aggregate metrics ---
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except Exception:
        mcc = 0.0

    try:
        kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    except Exception:
        kappa = 0.0

    # Top-2 Accuracy
    try:
        all_logits_t = torch.cat(all_logits, dim=0)  # [N, num_classes]
        top2_preds = all_logits_t.topk(min(2, all_logits_t.shape[1]), dim=-1).indices.numpy()
        top2_correct = sum(all_labels[i] in top2_preds[i] for i in range(len(all_labels)))
        top2_acc = top2_correct / max(len(all_labels), 1)
    except Exception:
        top2_acc = 0.0

    # --- Per-class metrics ---
    present_labels = sorted(set(all_labels))
    present_names = [level_labels[i] for i in present_labels]

    report_dict = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )

    per_class = {}
    for name in present_names:
        if name in report_dict:
            per_class[name] = {
                "precision": round(report_dict[name]["precision"], 4),
                "recall":    round(report_dict[name]["recall"], 4),
                "f1":        round(report_dict[name]["f1-score"], 4),
                "support":   int(report_dict[name]["support"]),
            }

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    # Normalize theo hàng
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    if verbose:
        print("\n[Classifier] Classification Report:")
        print(report_str)
        print(f"  Accuracy     : {accuracy:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  F1 (macro)   : {f1_macro:.4f}")
        print(f"  MCC          : {mcc:.4f}")
        print(f"  Kappa (κ)    : {kappa:.4f}")
        print(f"  Top-2 Acc    : {top2_acc:.4f}")
        print()

        # In Confusion Matrix
        try:
            max_len = max(max(len(n) for n in present_names), 12)
            header = f"{'True \\ Pred':<{max_len}} |" + "".join(f" {n:<{max_len}}" for n in present_names)
            print("[Classifier] Confusion Matrix (raw count):")
            print(header)
            print("-" * len(header))
            for i, row_name in enumerate(present_names):
                row_str = f"{row_name:<{max_len}} |" + "".join(f" {cm[i, j]:<{max_len}}" for j in range(len(present_names)))
                print(row_str)
            print()

            print("[Classifier] Confusion Matrix (normalized by row, %):")
            print(header)
            print("-" * len(header))
            for i, row_name in enumerate(present_names):
                row_str = f"{row_name:<{max_len}} |" + "".join(f" {cm_norm[i,j]*100:>{max_len}.1f}" for j in range(len(present_names)))
                print(row_str)
            print()
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể in confusion matrix: {e}")

    model.train()
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "top2_accuracy": top2_acc,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "all_preds": list(all_preds),
        "all_labels": list(all_labels),
        "present_labels": present_labels,
        "present_names": present_names,
        "classification_report_str": report_str,
    }



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
        early_stopping_patience = benchmark_cfg.get("early_stopping_patience", None)
        print(f"[{model_display_name}] Ordinal lambda_penalty: {lambda_penalty}")
        if early_stopping_patience:
            print(f"[{model_display_name}] Early stopping patience: {early_stopping_patience}")

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
        model = model.float().to(device)  # Ép kiểu float32 để tránh lỗi gradients FP16 khi dùng AMP trên T4

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

        # Mixed precision (sử dụng API torch.amp mới thay thế torch.cuda.amp bị deprecated)
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # --- Training Loop ---
        print(f"[{model_display_name}] Bắt đầu train ({num_epochs} epochs)...")
        best_f1 = -1.0
        best_metrics = {}
        best_model_dir = os.path.join(model_output_dir, "best_model")
        global_step = 0
        start_time = time.time()

        # Custom loss
        loss_fct = OrdinalLoss(class_weights=class_weights, level_labels=level_labels, lambda_penalty=lambda_penalty).to(device)

        # Lịch sử train (theo dõi loss và metrics từng epoch)
        train_history = []
        early_stop_counter = 0
        stop_training = False

        model.train()
        for epoch in range(1, num_epochs + 1):
            if stop_training:
                break
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
                    with torch.amp.autocast("cuda"):
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

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Evaluate cuối epoch (tắt verbose để không spam per-class mỗi epoch)
            print(f"\n[{model_display_name}] Evaluate sau Epoch {epoch}...")
            metrics = evaluate(model, val_loader, device, level_labels, loss_fct=loss_fct, verbose=False)
            print(
                f"  Train Loss={avg_train_loss:.4f} | "
                f"Val Loss={metrics['loss']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1w={metrics['f1_weighted']:.4f} | "
                f"F1m={metrics['f1_macro']:.4f} | "
                f"MCC={metrics['mcc']:.4f} | "
                f"Top2={metrics['top2_accuracy']:.4f}"
            )

            # Lưu lịch sử
            train_history.append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(metrics["loss"], 6),
                "val_accuracy": round(metrics["accuracy"], 6),
                "val_f1_weighted": round(metrics["f1_weighted"], 6),
                "val_f1_macro": round(metrics["f1_macro"], 6),
                "val_mcc": round(metrics["mcc"], 6),
                "val_kappa": round(metrics["cohen_kappa"], 6),
                "val_top2_acc": round(metrics["top2_accuracy"], 6),
            })

            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                best_metrics = metrics.copy()
                early_stop_counter = 0
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print(f"  *** Best! F1={best_f1:.4f} → Saved ***\n")
            else:
                early_stop_counter += 1
                patience_str = str(early_stopping_patience) if early_stopping_patience else "–"
                print(f"  [EarlyStopping] Không cải thiện ({early_stop_counter}/{patience_str})")
                if early_stopping_patience and early_stop_counter >= early_stopping_patience:
                    print(f"  [EarlyStopping] Dừng sớm tại epoch {epoch}!")
                    stop_training = True

            model.train()

        train_time_sec = time.time() - start_time
        train_time_str = format_time(train_time_sec)
        epochs_run = len(train_history)
        print(f"[{model_display_name}] Hoàn thành sau {train_time_str} ({epochs_run}/{num_epochs} epochs)")

        # Lưu training history
        history_path = os.path.join(model_output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, ensure_ascii=False, indent=2)
        print(f"  Lịch sử train → {history_path}")



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
        best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).float().to(device)
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

        # verbose=True để in đầy đủ per-class report + confusion matrix ra terminal
        test_metrics = evaluate(best_model, test_loader, device, level_labels, loss_fct=loss_fct, verbose=True)
        print("\n" + "="*70)
        print(f"  [{model_display_name}] KẾT QUẢ TRÊN TẬP TEST ĐỘC LẬP")
        print("="*70)
        print(f"  Test Loss         : {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy     : {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  Test F1 (macro)   : {test_metrics['f1_macro']:.4f}")
        print(f"  Test MCC          : {test_metrics['mcc']:.4f}")
        print(f"  Test Cohen Kappa  : {test_metrics['cohen_kappa']:.4f}")
        print(f"  Test Top-2 Acc    : {test_metrics['top2_accuracy']:.4f}")
        print("\n  === Per-class (Test Set) ===")
        for cls_name, cls_m in test_metrics.get("per_class", {}).items():
            print(f"  {cls_name:<12}: P={cls_m['precision']:.4f} | R={cls_m['recall']:.4f} | F1={cls_m['f1']:.4f} | Support={cls_m['support']}")
        print("="*70 + "\n")

        # Lưu báo cáo đầy đủ ra file
        import csv
        report_path = os.path.join(model_output_dir, "test_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_display_name} ({model_name})\n")
            f.write(f"Epochs run: {epochs_run}/{num_epochs}\n")
            f.write(f"Train time: {train_time_str}\n\n")
            f.write("=== Aggregate Metrics (Test Set) ===\n")
            f.write(f"  Loss         : {test_metrics['loss']:.6f}\n")
            f.write(f"  Accuracy     : {test_metrics['accuracy']:.6f}\n")
            f.write(f"  F1 (weighted): {test_metrics['f1_weighted']:.6f}\n")
            f.write(f"  F1 (macro)   : {test_metrics['f1_macro']:.6f}\n")
            f.write(f"  MCC          : {test_metrics['mcc']:.6f}\n")
            f.write(f"  Cohen Kappa  : {test_metrics['cohen_kappa']:.6f}\n")
            f.write(f"  Top-2 Acc    : {test_metrics['top2_accuracy']:.6f}\n\n")
            f.write("=== Classification Report ===\n")
            f.write(test_metrics.get("classification_report_str", "") + "\n")
            f.write("=== Training History ===\n")
            for h in train_history:
                f.write(json.dumps(h) + "\n")
        print(f"  Báo cáo đầy đủ → {report_path}")

        # Lưu confusion matrix (raw count)
        cm = test_metrics.get("confusion_matrix")
        present_names = test_metrics.get("present_names", level_labels)
        if cm is not None:
            cm_path = os.path.join(model_output_dir, "confusion_matrix.csv")
            with open(cm_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["True\\Pred"] + present_names)
                for i, row_name in enumerate(present_names):
                    writer.writerow([row_name] + [int(cm[i, j]) for j in range(len(present_names))])
            print(f"  Confusion matrix (raw) → {cm_path}")

            cm_norm = test_metrics.get("confusion_matrix_norm")
            if cm_norm is not None:
                cm_norm_path = os.path.join(model_output_dir, "confusion_matrix_normalized.csv")
                with open(cm_norm_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["True\\Pred"] + present_names)
                    for i, row_name in enumerate(present_names):
                        writer.writerow([row_name] + [f"{cm_norm[i, j]:.4f}" for j in range(len(present_names))])
                print(f"  Confusion matrix (norm) → {cm_norm_path}")

        result.update({
            "accuracy":       test_metrics.get("accuracy", 0.0),
            "f1_weighted":    test_metrics.get("f1_weighted", 0.0),
            "f1_macro":       test_metrics.get("f1_macro", 0.0),
            "mcc":            test_metrics.get("mcc", 0.0),
            "cohen_kappa":    test_metrics.get("cohen_kappa", 0.0),
            "top2_accuracy":  test_metrics.get("top2_accuracy", 0.0),
            "test_loss":      test_metrics.get("loss", 99.9),
            "per_class":      test_metrics.get("per_class", {}),
            "train_time":     train_time_str,
            "train_time_sec": train_time_sec,
            "epochs_run":     epochs_run,
            "model_dir":      best_model_dir,
            "n_params":       n_params,
            "status":         "SUCCESS",
        })

        # Giải phóng GPU memory
        del model
        del best_model
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
    Bao gồm: Accuracy, F1 (weighted), F1 (macro), MCC, Kappa, Top-2 Acc.
    Theo sau là bảng so sánh F1 từng class và bảng chi tiết Precision/Recall/F1/Support từng class.
    """
    print_banner("KẾT QUẢ BENCHMARK — TỔNG HỢP")

    # Bảng 1: Aggregate metrics
    col_widths = {
        "name": 18, "model_name": 30, "accuracy": 10,
        "f1_weighted": 10, "f1_macro": 10, "mcc": 8,
        "cohen_kappa": 8, "top2_accuracy": 10, "train_time": 12, "status": 8
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
        "accuracy": "Acc(Test)", "f1_weighted": "F1w(Test)",
        "f1_macro": "F1m(Test)", "mcc": "MCC",
        "cohen_kappa": "Kappa", "top2_accuracy": "Top2-Acc",
        "train_time": "Train Time", "status": "Status"
    }

    print(row_sep())
    print(row_data(headers))
    print(row_sep())

    sorted_results = sorted(
        results,
        key=lambda r: (r["status"] != "SUCCESS", -r.get("f1_weighted", 0))
    )

    for r in sorted_results:
        ok = r["status"] == "SUCCESS"
        display = {
            "name": r["name"],
            "model_name": r["model_name"],
            "accuracy":     f"{r.get('accuracy', 0):.4f}" if ok else "FAILED",
            "f1_weighted":  f"{r.get('f1_weighted', 0):.4f}" if ok else "FAILED",
            "f1_macro":     f"{r.get('f1_macro', 0):.4f}" if ok else "FAILED",
            "mcc":          f"{r.get('mcc', 0):.4f}" if ok else "FAILED",
            "cohen_kappa":  f"{r.get('cohen_kappa', 0):.4f}" if ok else "FAILED",
            "top2_accuracy":f"{r.get('top2_accuracy', 0):.4f}" if ok else "FAILED",
            "train_time":   r.get("train_time", "N/A"),
            "status":       r["status"],
        }
        print(row_data(display))

    print(row_sep())

    # Bảng 2: Per-class F1 summary
    level_labels_order = ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"]
    success_results = [r for r in results if r["status"] == "SUCCESS"]

    if success_results and any(r.get("per_class") for r in success_results):
        all_classes = sorted({
            cls for r in success_results
            for cls in r.get("per_class", {}).keys()
        }, key=lambda c: level_labels_order.index(c) if c in level_labels_order else 99)

        print("\n=== Per-class F1 trên Test Set ===")
        pc_widths = {"name": 18}
        for cls in all_classes:
            pc_widths[cls] = max(len(cls), 8)

        def pc_sep():
            return "+" + "+".join("-" * (w + 2) for w in pc_widths.values()) + "+"
        def pc_row(vals):
            cells = []
            for key, w in pc_widths.items():
                v = str(vals.get(key, "N/A"))
                cells.append(f" {v[:w]:<{w}} ")
            return "|" + "|".join(cells) + "|"

        pc_header = {"name": "Model"}
        for cls in all_classes:
            pc_header[cls] = f"F1-{cls}"
        print(pc_sep())
        print(pc_row(pc_header))
        print(pc_sep())
        for r in sorted_results:
            if r["status"] != "SUCCESS":
                continue
            row_vals = {"name": r["name"]}
            for cls in all_classes:
                cls_info = r.get("per_class", {}).get(cls, {})
                row_vals[cls] = f"{cls_info.get('f1', 0):.4f}" if cls_info else "N/A"
            print(pc_row(row_vals))
        print(pc_sep())

        # Bảng 3: Chi tiết metrics từng class cho mọi model
        print("\n=== Chi tiết metrics từng Class của các mô hình (Test Set) ===")
        pc_det_widths = {"model": 20, "class": 12, "precision": 10, "recall": 10, "f1": 10, "support": 8}
        def pcd_sep():
            return "+" + "+".join("-" * (w + 2) for w in pc_det_widths.values()) + "+"
        def pcd_row(vals):
            cells = []
            for key, w in pc_det_widths.items():
                v = str(vals.get(key, "N/A"))
                cells.append(f" {v[:w]:<{w}} ")
            return "|" + "|".join(cells) + "|"

        pcd_headers = {"model": "Model Name", "class": "Class", "precision": "Precision", "recall": "Recall", "f1": "F1-Score", "support": "Support"}
        print(pcd_sep())
        print(pcd_row(pcd_headers))
        print(pcd_sep())
        for r in sorted_results:
            if r["status"] != "SUCCESS":
                continue
            first_row = True
            for cls in all_classes:
                cls_info = r.get("per_class", {}).get(cls, {})
                if not cls_info:
                    continue
                row_vals = {
                    "model": r["name"] if first_row else "",
                    "class": cls,
                    "precision": f"{cls_info.get('precision', 0):.4f}",
                    "recall": f"{cls_info.get('recall', 0):.4f}",
                    "f1": f"{cls_info.get('f1', 0):.4f}",
                    "support": str(cls_info.get('support', 0)),
                }
                print(pcd_row(row_vals))
                first_row = False
            print(pcd_sep())

    # Winner
    if success_results:
        best = max(success_results, key=lambda r: r.get("f1_weighted", 0))
        best_macro = max(success_results, key=lambda r: r.get("f1_macro", 0))
        fastest = min(success_results, key=lambda r: r.get("train_time_sec", float("inf")))
        print(f"\n🏆 Best F1 (weighted): {best['name']} ({best['f1_weighted']:.4f})")
        print(f"🥇 Best F1 (macro)   : {best_macro['name']} ({best_macro['f1_macro']:.4f})")
        print(f"⚡ Fastest           : {fastest['name']} ({fastest['train_time']})")


# ----------------------------------------------------------------
# Lưu kết quả ra file
# ----------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Lưu kết quả benchmark ra 2 file:
    - benchmark_results.csv : CSV đầy đủ (dễ import vào Excel/Sheets)
    - benchmark_results.txt : Bảng ASCII (dễ đọc, có per-class summary)

    Args:
        results   : List kết quả benchmark
        output_dir: Thư mục output
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV với đầy đủ metrics ---
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = [
        "name", "model_name", "status",
        "accuracy", "f1_weighted", "f1_macro", "mcc", "cohen_kappa", "top2_accuracy",
        "test_loss", "train_time", "train_time_sec", "epochs_run",
        "n_params", "model_dir", "error"
    ]
    # Thêm per-class columns
    level_labels_order = ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"]
    all_classes = sorted({
        cls for r in results for cls in r.get("per_class", {}).keys()
    }, key=lambda c: level_labels_order.index(c) if c in level_labels_order else 99)

    for cls in all_classes:
        fieldnames += [f"{cls}_precision", f"{cls}_recall", f"{cls}_f1", f"{cls}_support"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            # Flatten float fields
            for k in ["accuracy", "f1_weighted", "f1_macro", "mcc", "cohen_kappa", "top2_accuracy", "test_loss"]:
                if isinstance(row.get(k), float):
                    row[k] = f"{row[k]:.6f}"
            # Flatten per_class
            for cls in all_classes:
                cls_info = r.get("per_class", {}).get(cls, {})
                row[f"{cls}_precision"] = f"{cls_info.get('precision', 0):.6f}" if cls_info else ""
                row[f"{cls}_recall"]    = f"{cls_info.get('recall', 0):.6f}" if cls_info else ""
                row[f"{cls}_f1"]        = f"{cls_info.get('f1', 0):.6f}" if cls_info else ""
                row[f"{cls}_support"]   = cls_info.get("support", "") if cls_info else ""
            # Xóa fields không phải string/int/float
            row.pop("per_class", None)
            row.pop("confusion_matrix", None)
            row.pop("confusion_matrix_norm", None)
            row.pop("all_preds", None)
            row.pop("all_labels", None)
            row.pop("present_labels", None)
            row.pop("present_names", None)
            row.pop("classification_report_str", None)
            writer.writerow(row)

    print(f"\n[Benchmark] Kết quả CSV: {csv_path}")

    # --- TXT (bảng ASCII đầy đủ) ---
    txt_path = os.path.join(output_dir, "benchmark_results.txt")
    lines = []
    lines.append("=" * 120)
    lines.append("BENCHMARK RESULTS — Level Classifier")
    lines.append("=" * 120)
    header_str = (
        f"{'Model':<20} {'HuggingFace ID':<33} "
        f"{'Acc':>8} {'F1w':>8} {'F1m':>8} {'MCC':>8} "
        f"{'Kappa':>8} {'Top2':>8} {'Loss':>10} {'Time':>12} {'Epochs':>8} {'Status'}"
    )
    lines.append(header_str)
    lines.append("-" * 120)

    sorted_r = sorted(results, key=lambda r: (r["status"] != "SUCCESS", -r.get("f1_weighted", 0)))
    for r in sorted_r:
        if r["status"] == "SUCCESS":
            ep_str = f"{r.get('epochs_run', '?')}"
            lines.append(
                f"{r['name']:<20} {r['model_name']:<33} "
                f"{r.get('accuracy', 0):>8.4f} {r.get('f1_weighted', 0):>8.4f} "
                f"{r.get('f1_macro', 0):>8.4f} {r.get('mcc', 0):>8.4f} "
                f"{r.get('cohen_kappa', 0):>8.4f} {r.get('top2_accuracy', 0):>8.4f} "
                f"{r.get('test_loss', 0):>10.4f} {r.get('train_time', 'N/A'):>12} "
                f"{ep_str:>8} OK"
            )
        else:
            lines.append(f"{r['name']:<20} {r['model_name']:<33} {'FAILED':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>10} {'N/A':>12} {'—':>8} FAILED")
            if r.get("error"):
                lines.append(f"  Error: {r['error'][:100]}")

    lines.append("=" * 120)

    # Per-class F1 summary
    success_r = [r for r in results if r["status"] == "SUCCESS" and r.get("per_class")]
    if success_r and all_classes:
        lines.append("\n=== Per-class F1 (Test Set) ===")
        pc_header = f"{'Model':<20}" + "".join(f" {cls:>12}" for cls in all_classes)
        lines.append(pc_header)
        lines.append("-" * (20 + 13 * len(all_classes)))
        for r in sorted_r:
            if r["status"] != "SUCCESS":
                continue
            row_str = f"{r['name']:<20}"
            for cls in all_classes:
                cls_f1 = r.get("per_class", {}).get(cls, {}).get("f1", None)
                row_str += f" {cls_f1:>12.4f}" if cls_f1 is not None else f" {'N/A':>12}"
            lines.append(row_str)
        lines.append("=" * (20 + 13 * len(all_classes)))

        # Chi tiết Precision/Recall/F1/Support từng class cho mọi model
        lines.append("\n=== Chi tiết metrics từng Class của các mô hình (Test Set) ===")
        header_pcd = f"{'Model':<20} {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}"
        lines.append(header_pcd)
        lines.append("-" * 78)
        for r in sorted_r:
            if r["status"] != "SUCCESS":
                continue
            first_row = True
            for cls in all_classes:
                cls_info = r.get("per_class", {}).get(cls, {})
                if not cls_info:
                    continue
                lines.append(
                    f"{r['name'] if first_row else '':<20} "
                    f"{cls:<12} "
                    f"{cls_info.get('precision', 0):>10.4f} "
                    f"{cls_info.get('recall', 0):>10.4f} "
                    f"{cls_info.get('f1', 0):>10.4f} "
                    f"{cls_info.get('support', 0):>8}"
                )
                first_row = False
            lines.append("-" * 78)

    # Winner summary
    if success_r:
        best = max(success_r, key=lambda r: r.get("f1_weighted", 0))
        lines.append(f"Best F1 (weighted): {best['name']} — F1={best['f1_weighted']:.4f}")
        best_macro = max(success_r, key=lambda r: r.get("f1_macro", 0))
        lines.append(f"Best F1 (macro)   : {best_macro['name']} — F1={best_macro['f1_macro']:.4f}")

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
