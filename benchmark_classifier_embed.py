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

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train MLP trên embedding vectors."""
        import torch
        from sklearn.metrics import f1_score

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        best_f1 = -1.0
        best_state = None

        print(f"  Train MLP head ({self.epochs} epochs, batch={self.batch_size})...")
        for epoch in range(1, self.epochs + 1):
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

            # Evaluate trên val set mỗi 10 epoch
            if X_val is not None and epoch % 10 == 0:
                val_metrics = self.score(X_val, y_val)
                if val_metrics["f1_weighted"] > best_f1:
                    best_f1 = val_metrics["f1_weighted"]
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"    [E{epoch:3d}/{self.epochs}] Loss={avg_loss:.4f} | Val F1={val_metrics['f1_weighted']:.4f} | Val Acc={val_metrics['accuracy']:.4f}")

        # Restore best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate MLP trên embedding vectors."""
        import torch
        from sklearn.metrics import accuracy_score, f1_score

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size * 4):
                xb = X_t[i:i + self.batch_size * 4]
                logits = self.model(xb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)

        acc = accuracy_score(y, all_preds)
        f1 = f1_score(y, all_preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1_weighted": f1}


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
        mlp.fit(train_embeds, train_labels, X_val=val_embeds, y_val=val_labels)

        train_time_sec = time.time() - start_time
        train_time_str = format_time(train_time_sec)

        # BƯỚC 3: Đánh giá trên tập Test
        print(f"\n[{model_display_name}] BƯỚC 3: Đánh giá trên TẬP TEST...")
        test_metrics = mlp.score(test_embeds, test_labels)
        print("=" * 60)
        print(f"  [{model_display_name}] KẾT QUẢ TRÊN TẬP TEST")
        print("=" * 60)
        print(f"  Test Accuracy   : {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  Tổng thời gian  : {train_time_str}")
        print("=" * 60 + "\n")

        result.update({
            "accuracy": test_metrics["accuracy"],
            "f1_weighted": test_metrics["f1_weighted"],
            "train_time": train_time_str,
            "train_time_sec": train_time_sec,
            "status": "SUCCESS",
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
    """In bảng so sánh kết quả benchmark embedding."""
    print_banner("KẾT QUẢ BENCHMARK — EMBEDDING + MLP HEAD")

    col_widths = {
        "name": 22, "model_name": 38, "accuracy": 12,
        "f1_weighted": 12, "train_time": 14, "embed_dim": 10, "status": 8
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
        "train_time": "Encode+Train", "embed_dim": "Embed Dim", "status": "Status"
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
            "train_time": r.get("train_time", "N/A"),
            "embed_dim": str(r.get("embed_dim", "N/A")),
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
    """Lưu kết quả ra CSV và TXT."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "benchmark_embed_results.csv")
    fieldnames = ["name", "model_name", "status", "accuracy", "f1_weighted",
                  "train_time", "train_time_sec", "embed_dim", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in ["accuracy", "f1_weighted"]:
                if isinstance(row.get(k), float):
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"\n[Benchmark Embed] CSV: {csv_path}")

    txt_path = os.path.join(output_dir, "benchmark_embed_results.txt")
    lines = ["=" * 100, "BENCHMARK EMBEDDING + MLP HEAD RESULTS — Level Classifier", "=" * 100]
    sorted_r = sorted(results, key=lambda r: (r["status"] != "SUCCESS", -r.get("f1_weighted", 0)))
    for r in sorted_r:
        if r["status"] == "SUCCESS":
            lines.append(
                f"{r['name']:<24} {r['model_name']:<40} "
                f"Acc={r.get('accuracy', 0):.4f} F1={r.get('f1_weighted', 0):.4f} "
                f"Time={r.get('train_time', 'N/A')} Dim={r.get('embed_dim', 'N/A')}"
            )
        else:
            lines.append(f"{r['name']:<24} {r['model_name']:<40} FAILED: {r.get('error', '')[:60]}")
    lines.append("=" * 100)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Benchmark Embed] TXT: {txt_path}")

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
