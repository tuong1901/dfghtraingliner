"""
benchmark_classifier_embed.py
-------------------------------
Benchmark Embedding + MLP Head cho bài toán Level Classification.

Phương pháp hoàn toàn khác BERT fine-tune và LLM LoRA:
  1. Encoder bị FROZEN (không cập nhật trọng số, không gradient)
  2. Chạy forward pass 1 LẦN DUY NHẤT để encode toàn bộ dataset
     → Lấy embedding vector (mean pooling hoặc [CLS] token)
  3. Train 1 MLP head nhỏ (2 lớp Linear) trên các embedding đó

Ưu điểm:
  - RẤT NHANH: encode 1 lần, train MLP không tốn kém
  - Đánh giá benchmark sơ bộ trước khi đầu tư fine-tune sâu
  - VRAM thấp (chỉ cần load encoder khi encode, sau đó giải phóng)

Sử dụng:
    python benchmark_classifier_embed.py [--config config_benchmark_embed.yaml]

Yêu cầu thư viện:
    pip install sentence-transformers

Hàm chính:
    - encode_dataset()               : Encode toàn bộ data thành numpy array
    - MLPClassifierHead              : MLP 2 lớp đơn giản
    - run_single_embed_benchmark()   : Pipeline cho 1 mô hình embedding
    - run_all_embed_benchmarks()     : Duyệt toàn bộ model list
    - print_results_table()          : In bảng so sánh
    - save_results()                 : Lưu kết quả ra file
"""

import os
import sys
import csv
import json
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_dataset, set_seed,
    print_banner, check_device, format_time
)


# ----------------------------------------------------------------
# Chuẩn bị text trước khi encode (head+tail truncation)
# ----------------------------------------------------------------
def prepare_text(text: str, tokenizer, max_length: int = 512,
                 strategy: str = "head+tail", e5_prefix: str = "") -> str:
    """
    Cắt ghép văn bản về độ dài phù hợp trước khi encode.
    Với e5_prefix: thêm "query: " trước text (yêu cầu của E5 models).

    Returns:
        Chuỗi văn bản đã xử lý (để đưa vào tokenizer của sentence-transformers)
    """
    # Không cần decode/re-encode, chỉ cắt ký tự thô là đủ cho sentence-transformers
    # (sentence-transformers tự quản lý truncation bên trong)
    result = text.strip()
    if e5_prefix:
        result = e5_prefix + result
    return result


# ----------------------------------------------------------------
# Encode toàn bộ dataset thành embedding vectors
# ----------------------------------------------------------------
def encode_dataset(
    model_name: str,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    use_e5_prefix: bool = False,
    device: str = "cuda",
) -> np.ndarray:
    """
    Load mô hình embedding, encode toàn bộ texts thành numpy array.
    Encoder bị FROZEN hoàn toàn — không train, không gradient.

    Args:
        model_name    : HuggingFace model ID
        texts         : List chuỗi văn bản cần encode
        batch_size    : Batch size khi inference
        max_length    : Độ dài tối đa token khi encode
        use_e5_prefix : True nếu dùng multilingual-e5 (cần prefix "query: ")
        device        : "cuda" hoặc "cpu"

    Returns:
        numpy array shape [N, embedding_dim]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Load embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, device=device)

    if use_e5_prefix:
        processed_texts = ["query: " + t.strip() for t in texts]
    else:
        processed_texts = [t.strip() for t in texts]

    print(f"  Encoding {len(texts):,} texts (batch_size={batch_size})...")
    start = time.time()
    embeddings = embed_model.encode(
        processed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize để cosine similarity = dot product
        device=device,
    )
    elapsed = format_time(time.time() - start)
    print(f"  Done! Shape: {embeddings.shape} | Thời gian encode: {elapsed}")

    # Giải phóng GPU memory ngay sau khi encode
    del embed_model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return embeddings


# ----------------------------------------------------------------
# MLP Classifier Head
# ----------------------------------------------------------------
class MLPClassifierHead:
    """
    MLP 2 lớp đơn giản train trên top embedding vectors (frozen encoder).

    Không dùng PyTorch DataLoader vì data đã ở dạng numpy array,
    đủ nhỏ để batch thủ công. Nhanh hơn nhiều so với DataLoader overhead.

    Args:
        input_dim   : Kích thước embedding vector đầu vào
        hidden_dim  : Số neuron lớp ẩn
        num_classes : Số nhãn phân loại
        dropout     : Tỉ lệ dropout
        lr          : Learning rate
        epochs      : Số epoch train MLP
        batch_size  : Batch size khi train
        device      : "cuda" hoặc "cpu"
        seed        : Random seed
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 6,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        device: str = "cuda",
        seed: int = 42,
    ):
        import torch
        import torch.nn as nn

        set_seed(seed)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr / 10
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def fit(self, X: np.ndarray, y: np.ndarray, level_labels: List[str], X_val: np.ndarray = None, y_val: np.ndarray = None, early_stopping_patience: int = None) -> List[Dict[str, Any]]:
        """Train MLP trên embedding vectors. Trả về lịch sử huấn luyện."""
        import torch

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        best_f1 = -1.0
        best_state = None
        
        train_history = []
        early_stop_counter = 0
        stop_training = False

        print(f"  Train MLP head ({self.epochs} epochs, batch={self.batch_size})...")
        for epoch in range(1, self.epochs + 1):
            if stop_training:
                break
                
            self.model.train()
            # Shuffle
            perm = torch.randperm(len(X_t))
            X_t = X_t[perm]
            y_t = y_t[perm]

            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i:i + self.batch_size]
                yb = y_t[i:i + self.batch_size]
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fct(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Evaluate trên val set mỗi epoch (MLP chạy siêu nhanh nên eval mỗi epoch là tối ưu)
            if X_val is not None and y_val is not None:
                val_metrics = self.score(X_val, y_val, level_labels, loss_fct=self.loss_fct, verbose=False)
                
                # Lưu lịch sử
                train_history.append({
                    "epoch": epoch,
                    "train_loss": round(avg_train_loss, 6),
                    "val_loss": round(val_metrics["loss"], 6),
                    "val_accuracy": round(val_metrics["accuracy"], 6),
                    "val_f1_weighted": round(val_metrics["f1_weighted"], 6),
                    "val_f1_macro": round(val_metrics["f1_macro"], 6),
                    "val_mcc": round(val_metrics["mcc"], 6),
                    "val_kappa": round(val_metrics["cohen_kappa"], 6),
                    "val_top2_acc": round(val_metrics["top2_accuracy"], 6),
                })

                if val_metrics["f1_weighted"] > best_f1:
                    best_f1 = val_metrics["f1_weighted"]
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stopping_patience and early_stop_counter >= early_stopping_patience:
                        print(f"    [EarlyStopping] Dừng sớm tại epoch {epoch}!")
                        stop_training = True

                if epoch % 10 == 0 or epoch == 1 or stop_training:
                    print(
                        f"    [E{epoch:3d}/{self.epochs}] TrainLoss={avg_train_loss:.4f} | "
                        f"ValLoss={val_metrics['loss']:.4f} | ValF1w={val_metrics['f1_weighted']:.4f} | "
                        f"ValMCC={val_metrics['mcc']:.4f} | ValTop2={val_metrics['top2_accuracy']:.4f}"
                    )

        # Restore best state
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        return train_history

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        level_labels: List[str],
        loss_fct=None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate MLP trên embedding vectors, trả về đầy đủ các metrics chi tiết."""
        return evaluate_embed(
            model=self.model,
            X=X,
            y=y,
            device=self.device,
            level_labels=level_labels,
            loss_fct=loss_fct,
            verbose=verbose,
            batch_size=self.batch_size * 4,
        )


# ----------------------------------------------------------------
# Hàm evaluate_embed dùng chung cho Embedding MLP
# ----------------------------------------------------------------
def evaluate_embed(
    model,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    level_labels: List[str],
    loss_fct=None,
    verbose: bool = True,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """
    Đánh giá mô hình MLP phân loại embeddings.
    Tính toán đầy đủ aggregate metrics + per-class metrics + confusion matrix.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (
        classification_report, f1_score, accuracy_score,
        confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    )

    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            yb = y_t[i:i + batch_size]

            logits = model(xb)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())
            all_logits.append(logits.cpu())

            if loss_fct is not None:
                loss = loss_fct(logits, yb)
                total_loss += loss.item()
                n_batches += 1

    # --- Aggregate metrics ---
    avg_loss = total_loss / max(n_batches, 1) if loss_fct is not None else 0.0
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
        print("\n[Embed-MLP-Classifier] Classification Report:")
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
            print("[Embed-MLP-Classifier] Confusion Matrix (raw count):")
            print(header)
            print("-" * len(header))
            for i, row_name in enumerate(present_names):
                row_str = f"{row_name:<{max_len}} |" + "".join(f" {cm[i, j]:<{max_len}}" for j in range(len(present_names)))
                print(row_str)
            print()

            print("[Embed-MLP-Classifier] Confusion Matrix (normalized by row, %):")
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
# Benchmark 1 mô hình embedding
# ----------------------------------------------------------------
def run_single_embed_benchmark(
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
    Encode + train MLP head + evaluate cho 1 mô hình embedding.

    Args:
        model_cfg   : Config của model này
        bench_cfg   : Config chung của benchmark_embed
        train_data  : List dict train samples
        val_data    : List dict val samples
        test_data   : List dict test samples
        level_labels: Danh sách nhãn level
        device      : "cuda" hoặc "cpu"
        seed        : Random seed

    Returns:
        Dict kết quả
    """
    model_display_name = model_cfg.get("name", model_cfg["model_name"])
    model_name = model_cfg["model_name"]
    print_banner(f"EMBED BENCHMARK: {model_display_name} ({model_name})")

    result = {
        "name": model_display_name,
        "model_name": model_name,
        "accuracy": 0.0,
        "f1_weighted": 0.0,
        "train_time": "N/A",
        "train_time_sec": 0.0,
        "embed_dim": 0,
        "status": "FAILED",
        "error": "",
    }

    try:
        set_seed(seed)

        batch_size = model_cfg.get("batch_size", 32)
        use_e5_prefix = model_cfg.get("use_e5_prefix", False)
        max_length = bench_cfg.get("max_length", 512)

        mlp_hidden = bench_cfg.get("mlp_hidden_dim", 256)
        mlp_dropout = bench_cfg.get("mlp_dropout", 0.2)
        mlp_epochs = bench_cfg.get("mlp_epochs", 50)
        mlp_lr = bench_cfg.get("mlp_lr", 1e-3)
        mlp_batch = bench_cfg.get("mlp_batch_size", 256)

        # Chuẩn bị texts và labels
        def extract(data):
            texts, labels = [], []
            for item in data:
                text = item.get("text", "").strip()
                level = str(item.get("level", "")).upper().strip()
                if not text or level not in level_labels:
                    continue
                texts.append(text)
                labels.append(level_labels.index(level))
            return texts, np.array(labels)

        train_texts, train_labels = extract(train_data)
        val_texts,   val_labels   = extract(val_data)
        test_texts,  test_labels  = extract(test_data)

        print(f"  Samples → Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

        start_time = time.time()

        # BƯỚC 1: Encode toàn bộ dataset (frozen encoder, 1 lần duy nhất)
        print(f"\n[{model_display_name}] BƯỚC 1: Encode dataset...")
        all_texts = train_texts + val_texts + test_texts
        all_embeds = encode_dataset(
            model_name=model_name,
            texts=all_texts,
            batch_size=batch_size,
            max_length=max_length,
            use_e5_prefix=use_e5_prefix,
            device=device,
        )

        n_train = len(train_texts)
        n_val   = len(val_texts)
        train_embeds = all_embeds[:n_train]
        val_embeds   = all_embeds[n_train:n_train + n_val]
        test_embeds  = all_embeds[n_train + n_val:]

        embed_dim = all_embeds.shape[1]
        result["embed_dim"] = embed_dim
        print(f"  Embedding dim: {embed_dim}")

        early_stopping_patience = bench_cfg.get("early_stopping_patience", None)

        # BƯỚC 2: Train MLP head trên embedding
        print(f"\n[{model_display_name}] BƯỚC 2: Train MLP classifier head...")
        mlp = MLPClassifierHead(
            input_dim=embed_dim,
            hidden_dim=mlp_hidden,
            num_classes=len(level_labels),
            dropout=mlp_dropout,
            lr=mlp_lr,
            epochs=mlp_epochs,
            batch_size=mlp_batch,
            device=device,
            seed=seed,
        )
        # fit bây giờ trả về train_history và nhận early_stopping_patience
        train_history = mlp.fit(
            train_embeds, train_labels,
            level_labels=level_labels,
            X_val=val_embeds, y_val=val_labels,
            early_stopping_patience=early_stopping_patience
        )

        train_time_sec = time.time() - start_time
        train_time_str = format_time(train_time_sec)
        epochs_run = len(train_history)

        output_base = bench_cfg.get("output_dir", "./outputs/benchmark_embed")
        safe_name = model_display_name.replace("/", "_").replace(" ", "_").replace("-", "_")
        model_output_dir = os.path.join(output_base, safe_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Lưu training history
        history_path = os.path.join(model_output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, ensure_ascii=False, indent=2)
        print(f"  Lịch sử train → {history_path}")

        # BƯỚC 3: Đánh giá trên tập Test
        print(f"\n[{model_display_name}] BƯỚC 3: Đánh giá trên TẬP TEST...")
        test_metrics = mlp.score(test_embeds, test_labels, level_labels=level_labels, loss_fct=mlp.loss_fct, verbose=True)
        print("=" * 70)
        print(f"  [{model_display_name}] KẾT QUẢ TRÊN TẬP TEST")
        print("=" * 70)
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
        print("=" * 70 + "\n")

        # Lưu báo cáo đầy đủ ra file
        report_path = os.path.join(model_output_dir, "test_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_display_name} ({model_name})\n")
            f.write(f"MLP Epochs run: {epochs_run}/{mlp_epochs}\n")
            f.write(f"Total time (encode+train): {train_time_str}\n\n")
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
            "status":         "SUCCESS",
        })

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
    In bảng so sánh kết quả benchmark dạng ASCII table đẹp cho Embedding + MLP.
    Bao gồm: Accuracy, F1 (weighted), F1 (macro), MCC, Kappa, Top-2 Acc.
    Theo sau là bảng so sánh F1 từng class và bảng chi tiết Precision/Recall/F1/Support từng class.
    """
    print_banner("KẾT QUẢ BENCHMARK EMBEDDING — TỔNG HỢP")

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
# Lưu kết quả
# ----------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Lưu kết quả benchmark ra 2 file:
    - benchmark_embed_results.csv : CSV đầy đủ (dễ import vào Excel/Sheets)
    - benchmark_embed_results.txt : Bảng ASCII (dễ đọc, có per-class summary)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV với đầy đủ metrics ---
    csv_path = os.path.join(output_dir, "benchmark_embed_results.csv")
    fieldnames = [
        "name", "model_name", "status",
        "accuracy", "f1_weighted", "f1_macro", "mcc", "cohen_kappa", "top2_accuracy",
        "test_loss", "train_time", "train_time_sec", "epochs_run",
        "embed_dim", "error"
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

    print(f"\n[Benchmark Embed] Kết quả CSV: {csv_path}")

    # --- TXT (bảng ASCII đầy đủ) ---
    txt_path = os.path.join(output_dir, "benchmark_embed_results.txt")
    lines = []
    lines.append("=" * 120)
    lines.append("BENCHMARK EMBEDDING + MLP HEAD RESULTS — Level Classifier")
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

    print(f"[Benchmark Embed] Kết quả TXT: {txt_path}")

    return csv_path, txt_path


# ----------------------------------------------------------------
# Chạy toàn bộ benchmark embedding
# ----------------------------------------------------------------
def run_all_embed_benchmarks(cfg: dict) -> List[Dict[str, Any]]:
    """
    Duyệt qua danh sách embedding model trong config, encode + train MLP + evaluate.

    Args:
        cfg: Dict config đã load từ config_benchmark_embed.yaml

    Returns:
        List kết quả của từng model
    """
    print_banner("BẮT ĐẦU BENCHMARK EMBEDDING + MLP HEAD — Level Classifier")

    bench_cfg = cfg.get("benchmark_embed", {})
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)

    set_seed(seed)
    device = check_device()

    level_labels = [lv.upper() for lv in bench_cfg.get("level_labels", [
        "INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"
    ])]

    # Load dataset
    print("[Benchmark Embed] Load dataset...")
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
    print(f"[Benchmark Embed] Split → Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    model_list = bench_cfg.get("models", [])
    enabled_models = [m for m in model_list if m.get("enabled", True)]
    print(f"\n[Benchmark Embed] Sẽ benchmark {len(enabled_models)}/{len(model_list)} model:")
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

        result = run_single_embed_benchmark(
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
    print(f"\n[Benchmark Embed] Tất cả model xong. Tổng thời gian: {total_time}")

    print_results_table(all_results)

    output_dir = bench_cfg.get("output_dir", "./outputs/benchmark_embed")
    save_results(all_results, output_dir)

    return all_results


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Benchmark Embedding + MLP Head cho Level Classification",
        epilog="""
Ví dụ:
  python benchmark_classifier_embed.py
  python benchmark_classifier_embed.py --config config_benchmark_embed.yaml
        """,
    )
    parser.add_argument("--config", type=str, default="config_benchmark_embed.yaml")
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
    bench_out = cfg.get("benchmark_embed", {}).get("output_dir", "./outputs/benchmark_embed")
    if not os.path.isabs(bench_out):
        cfg["benchmark_embed"]["output_dir"] = str(
            (Path(config_path).parent / bench_out).resolve()
        )

    run_all_embed_benchmarks(cfg)
