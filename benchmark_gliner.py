"""
benchmark_gliner.py
--------------------
So sánh các biến thể GLiNER architecture trên tập NER của Job Description.

Sử dụng:
    python benchmark_gliner.py [--config config.yaml]
    python train_master.py --benchmark-gliner

Pipeline cho mỗi model:
    1. Đánh giá Baseline  → F1 + nDCG@10 (model pretrained, CHƯA fine-tune)
    2. Fine-tune GLiNER   → train trên cleaned_dataset
    3. Đánh giá Post FT   → F1 + nDCG@10 (sau fine-tune)
    4. In bảng so sánh Baseline vs Post-FT

Sau benchmark, tự động chạy full pipeline trên model tốt nhất (best by post-FT F1).

Các kiến trúc GLiNER được benchmark (cấu hình trong config.yaml → benchmark_gliner.models):
  1. GLiNER-Small-v2.1    : DeBERTa-xsmall (22M params), nhanh nhất
  2. GLiNER-Medium-v2.1   : DeBERTa-base   (86M params), cân bằng ⭐
  3. GLiNER-Large-v2.1    : DeBERTa-large  (304M params), chính xác nhất
  4. GLiNER-Small-v2.5    : DeBERTa-v3-xsmall, data train mới hơn
  5. GLiNER-Medium-v2.5   : DeBERTa-v3-base, mới + cân bằng ⭐
  6. GLiNER-Multi-v0.1    : XLM-RoBERTa multilingual (Vi + En)

Thiết kế phù hợp với data:
  - Entity types: SKILL (span ngắn 1-5 word) + EXPERIENCE (span trung bình 5-15 word)
  - JD text dài → max_length=1024, GLiNER dùng sliding window
  - SKILL >> EXPERIENCE (imbalanced) → monitor F1 per-type
  - Threshold mặc định 0.5, có thể thay trong config

Metrics:
  - F1/Precision/Recall per entity type + Overall (exact char-level span match)
  - nDCG@10 (Normalized DCG) : đo quality ranking của predictions theo confidence score
      Giải thích nDCG@10 trong NER:
      - Với mỗi document, sort predictions theo score giảm dần
      - Lấy top-10, label mỗi pred: 1 nếu exact match với gold, 0 nếu không
      - DCG@10 = Σ rel_i / log2(i+2)
      - IDCG@10 = DCG của ranking lý tưởng (correct ở trước)
      - nDCG@10 = DCG / IDCG (nDCG=1.0 nghĩa là tất cả correct đều ở top)
      - Macro-average qua tất cả document
  - Training time, inference speed

Output:
  - Mỗi model: ./outputs/benchmark_gliner/{model_name}/
      baseline_metrics.json  : Metrics TRƯỚC fine-tune
      best_model/            : Checkpoint sau fine-tune
          eval_metrics.json  : Metrics SAU fine-tune
  - ./outputs/benchmark_gliner/benchmark_gliner_results.csv + .txt
  - ./outputs/benchmark_gliner/best_model_full_pipeline.txt (full analysis)

Hàm chính:
  - prepare_gliner_samples()         : Chuyển dataset sang GLiNER format
  - compute_ndcg_at_k()              : Tính nDCG@K từ ranked predictions
  - compute_ner_metrics()            : Tính F1/P/R per entity type (exact span)
  - evaluate_gliner()                : Inference + tính F1 + nDCG@10
  - run_single_gliner_benchmark()    : Baseline eval → fine-tune → post eval
  - run_best_model_full_pipeline()   : Pipeline đầy đủ cho model tốt nhất
  - run_all_gliner_benchmarks()      : Duyệt tất cả model, tổng hợp kết quả
  - print_gliner_results_table()     : Bảng so sánh ASCII (Baseline vs Post-FT)
  - save_gliner_results()            : Lưu CSV + TXT
"""

import os
import sys
import json
import csv
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_dataset, set_seed,
    print_banner, check_device, format_time
)
from train_gliner import prepare_gliner_samples


# ================================================================
# nDCG@K
# ================================================================
def compute_ndcg_at_k(
    predictions_with_scores: List[Dict],
    gold_labels: List[Dict],
    k: int = 10,
) -> float:
    """
    Tính nDCG@K cho 1 document đơn lẻ (char-level exact span match).

    nDCG (Normalized Discounted Cumulative Gain) đo chất lượng ranking:
    - predictions được sort theo confidence score giảm dần
    - Mỗi prediction được gán relevance=1 nếu là exact match với gold span,
      relevance=0 nếu sai
    - DCG@K = Σ_{i=1}^{K} rel_i / log2(i+1)
    - IDCG@K = DCG của ranking lý tưởng (tất cả gold đúng ở đầu)
    - nDCG@K = DCG@K / IDCG@K

    Ý nghĩa cho NER:
    - nDCG@10 = 1.0 → model tự tin nhất với những span đúng (gold span ở top)
    - nDCG@10 = 0.0 → model tự tin nhất với những span sai

    Args:
        predictions_with_scores : List[Dict] - [{start, end, label, score}, ...]
                                  CHƯA cần sort, hàm sẽ sort theo score
        gold_labels             : List[Dict] - [{start, end, label}, ...]
        k                       : Top-K để tính DCG (mặc định 10)

    Returns:
        float : nDCG@K trong [0.0, 1.0]
    """
    # Gold set: (start, end, label_upper)
    gold_set = {
        (g["start"], g["end"], g["label"].upper())
        for g in gold_labels
    }

    if not gold_set:
        # Không có gold entity → nDCG không xác định, trả về 1.0
        # (nếu model cũng không predict gì) hoặc 0.0 (nếu model predict sai)
        return 1.0 if not predictions_with_scores else 0.0

    # Sort predictions theo score giảm dần
    sorted_preds = sorted(
        predictions_with_scores,
        key=lambda p: p.get("score", 0.0),
        reverse=True
    )

    # Tính DCG@K
    dcg = 0.0
    for i, pred in enumerate(sorted_preds[:k]):
        span = (pred["start"], pred["end"], pred["label"].upper())
        rel = 1.0 if span in gold_set else 0.0
        # Vị trí i (0-indexed) → log2(i+2) để tránh log2(1)=0
        dcg += rel / math.log2(i + 2)

    # Tính IDCG@K: lý tưởng là tất cả gold đúng ở đầu
    n_relevant = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_ndcg_corpus(
    predictions_list: List[List[Dict]],
    gold_labels_list: List[List[Dict]],
    k: int = 10,
) -> float:
    """
    Tính macro-average nDCG@K qua tất cả document trong corpus.

    Args:
        predictions_list  : List[List[Dict]] - predictions của từng document
                            (mỗi dict: {start, end, label, score})
        gold_labels_list  : List[List[Dict]] - gold labels của từng document
        k                 : Top-K (mặc định 10)

    Returns:
        float : macro-average nDCG@K
    """
    scores = []
    for preds, golds in zip(predictions_list, gold_labels_list):
        s = compute_ndcg_at_k(preds, golds, k=k)
        scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


# ================================================================
# F1/P/R per entity type (exact span match)
# ================================================================
def compute_ner_metrics(
    predictions: List[List[Dict]],
    gold_labels: List[List[Dict]],
    entity_types: List[str],
) -> Dict[str, float]:
    """
    Tính Precision, Recall, F1 theo exact span match (char-level).
    Trả về metrics tổng hợp và per entity type.

    Exact match: span (start, end, label) phải khớp hoàn toàn.
    Được dùng thay vì GLiNER's built-in eval để có per-type detail.

    Args:
        predictions : List[List[Dict]] - mỗi sample là list {start, end, label, score}
        gold_labels : List[List[Dict]] - mỗi sample là list {start, end, label}
        entity_types: Danh sách các entity type cần đánh giá

    Returns:
        Dict chứa:
        {
            "overall_f1", "overall_precision", "overall_recall",
            "SKILL_f1", "SKILL_precision", "SKILL_recall", "SKILL_tp", ...
            "EXPERIENCE_f1", ...
        }
    """
    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)

    for preds, golds in zip(predictions, gold_labels):
        gold_set = {
            (g["start"], g["end"], g["label"].upper())
            for g in golds
        }
        pred_set = {
            (p["start"], p["end"], p["label"].upper())
            for p in preds
        }

        for span in pred_set:
            lbl = span[2]
            if span in gold_set:
                type_tp[lbl] += 1
            else:
                type_fp[lbl] += 1

        for span in gold_set:
            lbl = span[2]
            if span not in pred_set:
                type_fn[lbl] += 1

    results = {}
    all_tp = all_fp = all_fn = 0

    for etype in entity_types:
        etype = etype.upper()
        tp = type_tp[etype]
        fp = type_fp[etype]
        fn = type_fn[etype]

        all_tp += tp
        all_fp += fp
        all_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[f"{etype}_precision"] = prec
        results[f"{etype}_recall"]    = rec
        results[f"{etype}_f1"]        = f1
        results[f"{etype}_tp"]        = tp
        results[f"{etype}_fp"]        = fp
        results[f"{etype}_fn"]        = fn

    prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    results["overall_precision"] = prec
    results["overall_recall"]    = rec
    results["overall_f1"]        = f1

    return results


# ================================================================
# Evaluate: inference → F1 + nDCG@10
# ================================================================
def evaluate_gliner(
    model,
    val_samples: List[Dict],
    entity_types: List[str],
    threshold: float = 0.5,
    batch_size: int = 8,
    ndcg_ks: List[int] = None,
    low_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Chạy inference GLiNER trên val set rồi tính F1 + nDCG@K cho nhiều giá trị K.

    Dùng2 threshold:
    - threshold        : Lọc predictions cho tính F1 (strict, mặc định 0.5)
    - low_threshold    : Lọc predictions cho ranking nDCG (loose, mặc định 0.1)
                         Giữ nhiều candidates hơn để có đủ candidates cho nDCG@10

    Args:
        model          : GLiNER model
        val_samples    : List {text, entities}
        entity_types   : Danh sách entity type
        threshold      : Ngưỡng confidence cho F1
        batch_size     : Số sample/batch inference
        ndcg_ks        : List giá trị K để tính nDCG (mặc định [5, 10])
        low_threshold  : Ngưỡng thấp hơn cho ranking

    Returns:
        Dict metrics:
        {
            "overall_f1", "overall_precision", "overall_recall",
            "SKILL_f1", "EXPERIENCE_f1", ...
            "ndcg_at_5", "ndcg_at_10",   # tính cho tất cả K trong ndcg_ks
            "infer_speed", "infer_time",
        }
    """
    if ndcg_ks is None:
        ndcg_ks = [5, 10]

    all_preds_f1   = []
    all_preds_rank = []
    all_golds      = []

    texts              = [s["text"]     for s in val_samples]
    gold_entities_list = [s["entities"] for s in val_samples]

    start_t = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]

        # Inference với threshold cao (F1)
        try:
            batch_f1 = model.batch_predict_entities(
                batch_texts, entity_types, threshold=threshold
            )
        except AttributeError:
            batch_f1 = [
                model.predict_entities(t, entity_types, threshold=threshold)
                for t in batch_texts
            ]

        # Inference với threshold thấp (ranking / nDCG)
        try:
            batch_rank = model.batch_predict_entities(
                batch_texts, entity_types, threshold=low_threshold
            )
        except AttributeError:
            batch_rank = [
                model.predict_entities(t, entity_types, threshold=low_threshold)
                for t in batch_texts
            ]

        all_preds_f1.extend(batch_f1)
        all_preds_rank.extend(batch_rank)
        all_golds.extend(gold_entities_list[i: i + batch_size])

    infer_time  = time.time() - start_t
    infer_speed = len(texts) / max(infer_time, 1e-6)

    # Tính F1
    ner_metrics = compute_ner_metrics(all_preds_f1, all_golds, entity_types)

    # Tính nDCG@K cho tất cả K trong ndcg_ks
    ndcg_metrics = {}
    for k in ndcg_ks:
        ndcg_metrics[f"ndcg_at_{k}"] = compute_ndcg_corpus(
            all_preds_rank, all_golds, k=k
        )

    metrics = {
        **ner_metrics,
        **ndcg_metrics,
        "infer_speed": infer_speed,
        "infer_time":  infer_time,
        "threshold":   threshold,
    }
    return metrics


# ================================================================
# In metrics 1 lần đánh giá
# ================================================================
def _print_eval_metrics(label: str, metrics: Dict, entity_types: List[str], ndcg_ks: List[int] = None):
    """In kết quả evaluate đẹp ra terminal."""
    if ndcg_ks is None:
        ndcg_ks = [5, 10]

    f1   = metrics.get("overall_f1", 0)
    prec = metrics.get("overall_precision", 0)
    rec  = metrics.get("overall_recall", 0)
    ndcg_parts = " | ".join(
        f"nDCG@{k}={metrics.get(f'ndcg_at_{k}', 0):.4f}" for k in ndcg_ks
    )

    print(f"\n  [{label}]")
    print(f"  Overall  : P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | {ndcg_parts}")
    for et in entity_types:
        et_up = et.upper()
        print(f"  {et_up:<12}: P={metrics.get(et_up+'_precision',0):.4f} | "
              f"R={metrics.get(et_up+'_recall',0):.4f} | "
              f"F1={metrics.get(et_up+'_f1',0):.4f} "
              f"(TP={metrics.get(et_up+'_tp',0)}, "
              f"FP={metrics.get(et_up+'_fp',0)}, "
              f"FN={metrics.get(et_up+'_fn',0)})")
    spd = metrics.get("infer_speed", 0)
    if spd:
        print(f"  Infer    : {spd:.1f} samples/sec")


# ================================================================
# Benchmark 1 model: Baseline → Fine-tune → Post-eval
# ================================================================
def run_single_gliner_benchmark(
    model_cfg: dict,
    benchmark_cfg: dict,
    train_samples: List[Dict],
    val_samples: List[Dict],
    entity_types: List[str],
    seed: int,
) -> Dict[str, Any]:
    """
    Pipeline đầy đủ cho 1 GLiNER variant:
      1. Load pretrained model
      2. Đánh giá BASELINE (F1 + nDCG@10) → lưu baseline_metrics.json
      3. Fine-tune trên train_samples
      4. Đánh giá POST-FINETUNE (F1 + nDCG@10) → lưu eval_metrics.json
      5. In bảng so sánh Baseline vs Post-FT

    Args:
        model_cfg      : Config của model này {name, model_name, ...}
        benchmark_cfg  : Config chung benchmark
        train_samples  : List {text, entities} train
        val_samples    : List {text, entities} val
        entity_types   : ["SKILL", "EXPERIENCE"]
        seed           : Random seed

    Returns:
        Dict kết quả đầy đủ với cả baseline và post-finetune metrics:
        {
            "name", "model_name", "status",
            "n_params",
            # Baseline (pretrained, no fine-tune)
            "baseline_f1", "baseline_ndcg_at_10",
            "baseline_SKILL_f1", "baseline_EXPERIENCE_f1", ...
            # Post fine-tune
            "overall_f1", "overall_precision", "overall_recall",
            "ndcg_at_10",
            "SKILL_f1", "EXPERIENCE_f1", ...
            # Delta
            "delta_f1", "delta_ndcg",
            # Meta
            "train_time", "train_time_sec", "infer_speed", "model_dir",
        }
    """
    model_display = model_cfg.get("name", model_cfg["model_name"])
    model_hf_id   = model_cfg["model_name"]
    ndcg_k        = benchmark_cfg.get("ndcg_k", 10)
    eval_threshold = benchmark_cfg.get("eval_threshold", 0.5)
    low_threshold  = benchmark_cfg.get("low_threshold", 0.1)
    eval_batch     = model_cfg.get("eval_batch_size", benchmark_cfg.get("eval_batch_size", 8))

    print_banner(f"BENCHMARK: {model_display}")
    print(f"  HuggingFace ID : {model_hf_id}")
    print(f"  Entity types   : {entity_types}")
    print(f"  Pipeline       : Baseline → Fine-tune → Post-eval (F1 + nDCG@{ndcg_k})")

    # --- Init result dict ---
    result = {
        "name": model_display,
        "model_name": model_hf_id,
        "status": "FAILED",
        "error": "",
        "n_params": 0,
        # Baseline
        "baseline_f1": 0.0,
        f"baseline_ndcg_at_{ndcg_k}": 0.0,
        # Post-FT
        "overall_f1": 0.0,
        "overall_precision": 0.0,
        "overall_recall": 0.0,
        f"ndcg_at_{ndcg_k}": 0.0,
        # Delta
        "delta_f1": 0.0,
        f"delta_ndcg_at_{ndcg_k}": 0.0,
        # Meta
        "train_time": "N/A",
        "train_time_sec": 0.0,
        "infer_speed": 0.0,
        "model_dir": "N/A",
    }
    for et in entity_types:
        et_up = et.upper()
        result[f"baseline_{et_up}_f1"] = 0.0
        result[f"{et_up}_f1"]          = 0.0
        result[f"{et_up}_precision"]   = 0.0
        result[f"{et_up}_recall"]      = 0.0

    try:
        from gliner import GLiNER
        from gliner.training import Trainer, TrainingArguments
        from gliner.data_processing.collator import DataCollator
        import torch

        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Hyperparameters ---
        num_epochs  = model_cfg.get("num_epochs", benchmark_cfg.get("num_epochs", 3))
        max_length  = model_cfg.get("max_length", benchmark_cfg.get("max_length", 1024))
        train_batch = model_cfg.get("train_batch_size", benchmark_cfg.get("train_batch_size", 8))
        lr          = model_cfg.get("learning_rate", benchmark_cfg.get("learning_rate", 5e-5))
        others_lr   = model_cfg.get("others_lr", benchmark_cfg.get("others_lr", 1e-5))
        weight_decay= model_cfg.get("weight_decay", benchmark_cfg.get("weight_decay", 0.01))
        warmup_ratio= model_cfg.get("warmup_ratio", benchmark_cfg.get("warmup_ratio", 0.1))
        grad_accum  = model_cfg.get("gradient_accumulation_steps",
                                    benchmark_cfg.get("gradient_accumulation_steps", 2))
        logging_steps = benchmark_cfg.get("logging_steps", 100)

        # Output dirs
        output_base = benchmark_cfg.get("output_dir", "./outputs/benchmark_gliner")
        safe_name   = model_display.replace("/","_").replace(" ","_").replace("-","_")
        model_output_dir = os.path.join(output_base, safe_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # ============================================================
        # STEP 1: Load pretrained model
        # ============================================================
        print(f"\n[{model_display}] STEP 1: Load pretrained model từ HuggingFace...")
        model = GLiNER.from_pretrained(model_hf_id)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params:,}")
        result["n_params"] = n_params

        # ============================================================
        # STEP 2: Đánh giá BASELINE (trước fine-tune)
        # ============================================================
        print(f"\n[{model_display}] STEP 2: Đánh giá BASELINE (pretrained, chưa fine-tune)...")
        print(f"  Threshold F1={eval_threshold} | nDCG low_threshold={low_threshold}")

        baseline_metrics = evaluate_gliner(
            model, val_samples, entity_types,
            threshold=eval_threshold,
            batch_size=eval_batch,
            ndcg_k=ndcg_k,
            low_threshold=low_threshold,
        )

        _print_eval_metrics("BASELINE", baseline_metrics, entity_types, ndcg_k)

        # Lưu baseline metrics
        baseline_path = os.path.join(model_output_dir, "baseline_metrics.json")
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": model_hf_id,
                "entity_types": entity_types,
                "stage": "baseline",
                **baseline_metrics,
            }, f, ensure_ascii=False, indent=2)

        # Cập nhật result với baseline
        result["baseline_f1"] = baseline_metrics.get("overall_f1", 0.0)
        result[f"baseline_ndcg_at_{ndcg_k}"] = baseline_metrics.get(f"ndcg_at_{ndcg_k}", 0.0)
        for et in entity_types:
            et_up = et.upper()
            result[f"baseline_{et_up}_f1"] = baseline_metrics.get(f"{et_up}_f1", 0.0)

        # ============================================================
        # STEP 3: Fine-tune
        # ============================================================
        print(f"\n[{model_display}] STEP 3: Fine-tune ({num_epochs} epochs)...")
        print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")
        print(f"  max_length={max_length} | batch={train_batch} | lr={lr:.1e} | grad_accum={grad_accum}")

        # Reload để fine-tune fresh (không bị ảnh hưởng bởi eval mode)
        model = GLiNER.from_pretrained(model_hf_id)

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            learning_rate=lr,
            weight_decay=weight_decay,
            others_lr=others_lr,
            others_weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_ratio=warmup_ratio,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=logging_steps,
            seed=seed,
            dataloader_num_workers=0,
            fp16=(device == "cuda"),
            gradient_accumulation_steps=grad_accum,
            report_to="none",
        )

        data_collator = DataCollator(
            model.config,
            data_processor=model.data_processor,
            prepare_labels=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_samples,
            eval_dataset=val_samples,
            tokenizer=model.data_processor.transformer_tokenizer,
            data_collator=data_collator,
        )

        start_t = time.time()
        trainer.train()
        train_time_sec = time.time() - start_t
        train_time_str = format_time(train_time_sec)
        print(f"[{model_display}] Fine-tune xong sau {train_time_str}")

        # Save best model
        best_dir = os.path.join(model_output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        with open(os.path.join(best_dir, "entity_types.json"), "w", encoding="utf-8") as f:
            json.dump({"entity_types": entity_types}, f, ensure_ascii=False, indent=2)
        print(f"  Model lưu: {best_dir}")

        # ============================================================
        # STEP 4: Đánh giá POST-FINETUNE
        # ============================================================
        print(f"\n[{model_display}] STEP 4: Đánh giá POST-FINETUNE...")

        ft_model = GLiNER.from_pretrained(best_dir)
        ft_model.eval()

        ft_metrics = evaluate_gliner(
            ft_model, val_samples, entity_types,
            threshold=eval_threshold,
            batch_size=eval_batch,
            ndcg_k=ndcg_k,
            low_threshold=low_threshold,
        )

        _print_eval_metrics("POST FINE-TUNE", ft_metrics, entity_types, ndcg_k)

        # Save post-FT metrics
        ft_metrics_path = os.path.join(best_dir, "eval_metrics.json")
        with open(ft_metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": model_hf_id,
                "entity_types": entity_types,
                "stage": "post_finetune",
                "n_params": n_params,
                "train_time": train_time_str,
                **ft_metrics,
            }, f, ensure_ascii=False, indent=2)

        # ============================================================
        # STEP 5: In bảng so sánh Baseline vs Post-FT
        # ============================================================
        delta_f1   = ft_metrics["overall_f1"]    - baseline_metrics["overall_f1"]
        delta_ndcg = ft_metrics.get(f"ndcg_at_{ndcg_k}", 0) - baseline_metrics.get(f"ndcg_at_{ndcg_k}", 0)

        print(f"\n[{model_display}] ── So sánh Baseline vs Post-FT ──")
        print(f"  {'Metric':<22} {'Baseline':>10} {'Post-FT':>10} {'Delta':>10}")
        print(f"  {'-'*52}")
        print(f"  {'Overall F1':<22} {baseline_metrics['overall_f1']:>10.4f} "
              f"{ft_metrics['overall_f1']:>10.4f} "
              f"{delta_f1:>+10.4f}")
        print(f"  {f'nDCG@{ndcg_k}':<22} "
              f"{baseline_metrics.get(f'ndcg_at_{ndcg_k}',0):>10.4f} "
              f"{ft_metrics.get(f'ndcg_at_{ndcg_k}',0):>10.4f} "
              f"{delta_ndcg:>+10.4f}")
        for et in entity_types:
            et_up = et.upper()
            b_f1 = baseline_metrics.get(f"{et_up}_f1", 0)
            f_f1 = ft_metrics.get(f"{et_up}_f1", 0)
            print(f"  {et_up+' F1':<22} {b_f1:>10.4f} {f_f1:>10.4f} {f_f1-b_f1:>+10.4f}")
        print(f"  {'-'*52}")

        # Update result
        result.update({
            "status": "SUCCESS",
            "overall_f1": ft_metrics["overall_f1"],
            "overall_precision": ft_metrics["overall_precision"],
            "overall_recall": ft_metrics["overall_recall"],
            f"ndcg_at_{ndcg_k}": ft_metrics.get(f"ndcg_at_{ndcg_k}", 0.0),
            "delta_f1": delta_f1,
            f"delta_ndcg_at_{ndcg_k}": delta_ndcg,
            "train_time": train_time_str,
            "train_time_sec": train_time_sec,
            "infer_speed": ft_metrics.get("infer_speed", 0.0),
            "model_dir": best_dir,
        })
        for et in entity_types:
            et_up = et.upper()
            result[f"{et_up}_f1"]        = ft_metrics.get(f"{et_up}_f1", 0.0)
            result[f"{et_up}_precision"] = ft_metrics.get(f"{et_up}_precision", 0.0)
            result[f"{et_up}_recall"]    = ft_metrics.get(f"{et_up}_recall", 0.0)

        # Dọn bộ nhớ
        del model, ft_model, trainer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        import traceback
        print(f"\n[{model_display}] LỖI: {e}")
        print(traceback.format_exc()[:600])
        result["error"] = str(e)
        result["status"] = "FAILED"

    return result


# ================================================================
# Full pipeline cho model tốt nhất (sau benchmark)
# ================================================================
def run_best_model_full_pipeline(
    best_result: Dict,
    benchmark_cfg: dict,
    train_samples: List[Dict],
    val_samples: List[Dict],
    entity_types: List[str],
    seed: int,
    full_epochs: int = 5,
) -> Dict[str, Any]:
    """
    Chạy full pipeline đầy đủ cho model tốt nhất (sau benchmark):
      1. Load model tốt nhất đã fine-tune (từ benchmark)
      2. Thêm 1 vòng đánh giá bằng nhiều threshold (0.3, 0.4, 0.5, 0.6)
      3. Nếu config yêu cầu, train thêm để đạt full_epochs
      4. Lưu báo cáo đầy đủ

    Hàm này nhận best_result từ run_all_gliner_benchmarks() thay vì train lại từ đầu,
    để tiết kiệm thời gian.

    Args:
        best_result   : Dict kết quả của model tốt nhất (từ benchmark)
        benchmark_cfg : Config benchmark
        train_samples : Train samples
        val_samples   : Val samples
        entity_types  : Entity types
        seed          : Seed
        full_epochs   : Số epoch đầy đủ nếu cần train thêm

    Returns:
        Dict kết quả pipeline đầy đủ
    """
    print_banner(f"FULL PIPELINE - BEST MODEL: {best_result['name']}")
    print(f"  Model     : {best_result['model_name']}")
    print(f"  Model dir : {best_result['model_dir']}")
    print(f"  Post-FT F1: {best_result['overall_f1']:.4f}")

    ndcg_k       = benchmark_cfg.get("ndcg_k", 10)
    output_dir   = benchmark_cfg.get("output_dir", "./outputs/benchmark_gliner")
    eval_batch   = benchmark_cfg.get("eval_batch_size", 8)
    low_threshold = benchmark_cfg.get("low_threshold", 0.1)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"FULL PIPELINE REPORT - BEST MODEL: {best_result['name']}")
    report_lines.append(f"Model ID  : {best_result['model_name']}")
    report_lines.append(f"n_params  : {best_result.get('n_params', 'N/A'):,}" if isinstance(
        best_result.get('n_params'), int) else f"n_params  : N/A")
    report_lines.append("=" * 80)

    try:
        from gliner import GLiNER

        # --- Load fine-tuned model ---
        model_dir = best_result["model_dir"]
        if not os.path.exists(model_dir):
            print(f"[LỖI] Không tìm thấy model dir: {model_dir}")
            return best_result

        print(f"\nLoad fine-tuned model từ: {model_dir}")
        model = GLiNER.from_pretrained(model_dir)
        model.eval()

        # --- Baseline (pretrained) đã có từ benchmark, load lại ---
        baseline_path = os.path.join(
            os.path.dirname(model_dir), "baseline_metrics.json"
        )
        baseline_metrics = {}
        if os.path.exists(baseline_path):
            with open(baseline_path, encoding="utf-8") as f:
                baseline_metrics = json.load(f)
            print(f"Đã load baseline metrics từ: {baseline_path}")
        else:
            print("[CẢNH BÁO] Không tìm thấy baseline_metrics.json, sẽ bỏ qua phần này")

        # --- Đánh giá ở nhiều threshold ---
        thresholds = [0.3, 0.4, 0.5, 0.6]
        report_lines.append("\n── Sensitivity Analysis: Multiple Thresholds ──")
        report_lines.append(
            f"{'Threshold':>12} {'F1':>8} {'P':>8} {'R':>8} "
            + " ".join(f"{et.upper()[:6]+'_F1':>10}" for et in entity_types)
            + f" {'nDCG@'+str(ndcg_k):>10}"
        )
        report_lines.append("-" * 80)

        threshold_results = []
        for thr in thresholds:
            m = evaluate_gliner(
                model, val_samples, entity_types,
                threshold=thr,
                batch_size=eval_batch,
                ndcg_k=ndcg_k,
                low_threshold=low_threshold,
            )
            threshold_results.append((thr, m))
            line = (
                f"{thr:>12.1f} {m['overall_f1']:>8.4f} "
                f"{m['overall_precision']:>8.4f} {m['overall_recall']:>8.4f}"
            )
            for et in entity_types:
                line += f" {m.get(et.upper()+'_f1', 0):>10.4f}"
            line += f" {m.get(f'ndcg_at_{ndcg_k}', 0):>10.4f}"
            print(f"  threshold={thr:.1f}: F1={m['overall_f1']:.4f} | "
                  f"nDCG@{ndcg_k}={m.get(f'ndcg_at_{ndcg_k}', 0):.4f}")
            report_lines.append(line)

        # Best threshold
        best_thr, best_thr_m = max(threshold_results, key=lambda x: x[1]["overall_f1"])
        report_lines.append(f"\n→ Best threshold by F1: {best_thr:.1f} (F1={best_thr_m['overall_f1']:.4f})")

        # --- Báo cáo so sánh Baseline vs Post-FT ---
        if baseline_metrics:
            report_lines.append("\n── Baseline vs Post-Fine-tune ──")
            report_lines.append(
                f"{'Metric':<22} {'Baseline':>10} {'Post-FT':>10} {'Delta':>10}"
            )
            report_lines.append("-" * 52)

            # Lấy post-FT tại threshold 0.5 (mặc định)
            post_ft_m = dict(threshold_results[[t[0] for t in threshold_results].index(0.5)][1]) if 0.5 in thresholds else {}

            for label, b_val, f_val in [
                ("Overall F1",
                 baseline_metrics.get("overall_f1", 0),
                 post_ft_m.get("overall_f1", best_result["overall_f1"])),
                (f"nDCG@{ndcg_k}",
                 baseline_metrics.get(f"ndcg_at_{ndcg_k}", 0),
                 post_ft_m.get(f"ndcg_at_{ndcg_k}", best_result.get(f"ndcg_at_{ndcg_k}", 0))),
            ]:
                delta = f_val - b_val
                report_lines.append(f"{label:<22} {b_val:>10.4f} {f_val:>10.4f} {delta:>+10.4f}")
            for et in entity_types:
                et_up = et.upper()
                b_val = baseline_metrics.get(f"{et_up}_f1", 0)
                f_val = post_ft_m.get(f"{et_up}_f1", best_result.get(f"{et_up}_f1", 0))
                delta = f_val - b_val
                report_lines.append(f"{et_up+' F1':<22} {b_val:>10.4f} {f_val:>10.4f} {delta:>+10.4f}")

        # --- Lưu báo cáo ---
        report_path = os.path.join(output_dir, "best_model_full_pipeline.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n[Full Pipeline] Báo cáo đầy đủ: {report_path}")

        # Lưu JSON
        json_path = os.path.join(output_dir, "best_model_full_pipeline.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_model": best_result,
                "threshold_analysis": [
                    {"threshold": t, **m} for t, m in threshold_results
                ],
                "best_threshold": best_thr,
                "baseline_metrics": baseline_metrics,
            }, f, ensure_ascii=False, indent=2)
        print(f"[Full Pipeline] JSON đầy đủ: {json_path}")

        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        import traceback
        print(f"\n[Full Pipeline] LỖI: {e}")
        print(traceback.format_exc()[:400])

    return best_result


# ================================================================
# In bảng kết quả benchmark (Baseline vs Post-FT)
# ================================================================
def print_gliner_results_table(results: List[Dict], entity_types: List[str], ndcg_k: int = 10):
    """
    In bảng so sánh kết quả benchmark GLiNER.
    Mỗi model có 2 hàng: Baseline và Post-FT với Delta.

    Args:
        results     : List dict kết quả từ run_single_gliner_benchmark()
        entity_types: Danh sách entity type
        ndcg_k      : K cho nDCG
    """
    print_banner("KẾT QUẢ BENCHMARK GLiNER")

    sorted_r = sorted(
        results,
        key=lambda r: (r["status"] != "SUCCESS", -r.get("overall_f1", 0))
    )

    sep = "-" * 90
    header = (f"{'Model':<24} {'Stage':<8} {'F1':>7} {'P':>7} {'R':>7} "
              f"{f'nDCG@{ndcg_k}':>9}")
    for et in entity_types:
        header += f" {et.upper()[:6]+'_F1':>10}"
    header += f"  {'Time':>9}"

    print(header)
    print(sep)

    for r in sorted_r:
        if r["status"] != "SUCCESS":
            print(f"{r['name']:<24} {'FAILED':<8} {'':>7} {'':>7} {'':>7} {'':>9}"
                  f"  {r.get('error','')[:30]}")
            continue

        # Hàng Baseline
        b_f1   = r.get("baseline_f1", 0)
        b_ndcg = r.get(f"baseline_ndcg_at_{ndcg_k}", 0)
        base_line = f"{r['name']:<24} {'BASE':<8} {b_f1:>7.4f} {'--':>7} {'--':>7} {b_ndcg:>9.4f}"
        for et in entity_types:
            base_line += f" {r.get('baseline_'+et.upper()+'_f1', 0):>10.4f}"
        base_line += f"  {'':>9}"
        print(base_line)

        # Hàng Post-FT
        f1   = r.get("overall_f1", 0)
        prec = r.get("overall_precision", 0)
        rec  = r.get("overall_recall", 0)
        ndcg = r.get(f"ndcg_at_{ndcg_k}", 0)
        d_f1 = r.get("delta_f1", 0)

        ft_line = (
            f"{'':24} {'POST-FT':<8} {f1:>7.4f} {prec:>7.4f} {rec:>7.4f} {ndcg:>9.4f}"
        )
        for et in entity_types:
            ft_line += f" {r.get(et.upper()+'_f1', 0):>10.4f}"
        ft_line += f"  {r.get('train_time', 'N/A'):>9}"
        print(ft_line)

        # Hàng Delta
        d_ndcg = r.get(f"delta_ndcg_at_{ndcg_k}", 0)
        delta_line = f"{'':24} {'Δ':<8} {d_f1:>+7.4f} {'':>7} {'':>7} {d_ndcg:>+9.4f}"
        for et in entity_types:
            d = r.get(f"{et.upper()}_f1", 0) - r.get(f"baseline_{et.upper()}_f1", 0)
            delta_line += f" {d:>+10.4f}"
        delta_line += f"  {r.get('infer_speed', 0):>7.0f}/s"
        print(delta_line)
        print(sep)

    # Tìm winners
    success = [r for r in results if r["status"] == "SUCCESS"]
    if success:
        best_f1    = max(success, key=lambda r: r.get("overall_f1", 0))
        best_ndcg  = max(success, key=lambda r: r.get(f"ndcg_at_{ndcg_k}", 0))
        best_delta = max(success, key=lambda r: r.get("delta_f1", 0))
        fastest    = min(success, key=lambda r: r.get("train_time_sec", float("inf")))

        print(f"\n🏆 Best Post-FT F1   : {best_f1['name']} (F1={best_f1['overall_f1']:.4f})")
        print(f"📈 Best Post-FT nDCG : {best_ndcg['name']} (nDCG@{ndcg_k}={best_ndcg.get(f'ndcg_at_{ndcg_k}', 0):.4f})")
        print(f"🚀 Most Improved F1  : {best_delta['name']} (ΔF1={best_delta.get('delta_f1', 0):+.4f})")
        print(f"⚡ Fastest train     : {fastest['name']} ({fastest['train_time']})")


# ================================================================
# Lưu kết quả ra CSV + TXT
# ================================================================
def save_gliner_results(
    results: List[Dict],
    entity_types: List[str],
    output_dir: str,
    ndcg_k: int = 10,
) -> Tuple[str, str]:
    """
    Lưu kết quả benchmark ra CSV và TXT.
    Mỗi row CSV có đủ cả Baseline và Post-FT metrics.

    Args:
        results     : List kết quả benchmark
        entity_types: List entity type
        output_dir  : Thư mục output
        ndcg_k      : K cho nDCG

    Returns:
        (csv_path, txt_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Xây dựng fieldnames ---
    base_fields = ["name", "model_name", "status", "n_params"]
    # Baseline fields
    baseline_fields = ["baseline_f1", f"baseline_ndcg_at_{ndcg_k}"]
    for et in entity_types:
        baseline_fields.append(f"baseline_{et.upper()}_f1")
    # Post-FT fields
    postft_fields = ["overall_f1", "overall_precision", "overall_recall",
                     f"ndcg_at_{ndcg_k}", "delta_f1", f"delta_ndcg_at_{ndcg_k}"]
    for et in entity_types:
        postft_fields += [f"{et.upper()}_f1", f"{et.upper()}_precision", f"{et.upper()}_recall"]
    # Meta
    meta_fields = ["train_time", "train_time_sec", "infer_speed", "model_dir", "error"]

    all_fields = base_fields + baseline_fields + postft_fields + meta_fields

    # --- CSV ---
    csv_path = os.path.join(output_dir, "benchmark_gliner_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in row:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"[Benchmark GLiNER] CSV: {csv_path}")

    # --- TXT ---
    txt_path = os.path.join(output_dir, "benchmark_gliner_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"BENCHMARK GLiNER - BASELINE vs POST-FINETUNE\n")
        f.write(f"Entity types : {entity_types}\n")
        f.write(f"Metrics      : F1 (exact span match, char-level) + nDCG@{ndcg_k}\n")
        f.write("=" * 100 + "\n\n")

        sorted_r = sorted(results, key=lambda r: (r["status"] != "SUCCESS", -r.get("overall_f1", 0)))

        for r in sorted_r:
            f.write(f"Model: {r['name']} ({r['model_name']})\n")
            if r["status"] != "SUCCESS":
                f.write(f"  STATUS: FAILED | Error: {r.get('error', '')[:80]}\n\n")
                continue

            b_f1   = r.get("baseline_f1", 0)
            b_ndcg = r.get(f"baseline_ndcg_at_{ndcg_k}", 0)
            f_f1   = r.get("overall_f1", 0)
            f_ndcg = r.get(f"ndcg_at_{ndcg_k}", 0)

            f.write(f"  {'Metric':<22} {'Baseline':>10} {'Post-FT':>10} {'Delta':>10}\n")
            f.write(f"  {'-'*52}\n")
            f.write(f"  {'Overall F1':<22} {b_f1:>10.4f} {f_f1:>10.4f} {f_f1-b_f1:>+10.4f}\n")
            f.write(f"  {f'nDCG@{ndcg_k}':<22} {b_ndcg:>10.4f} {f_ndcg:>10.4f} {f_ndcg-b_ndcg:>+10.4f}\n")
            for et in entity_types:
                et_up = et.upper()
                b = r.get(f"baseline_{et_up}_f1", 0)
                v = r.get(f"{et_up}_f1", 0)
                f.write(f"  {et_up+' F1':<22} {b:>10.4f} {v:>10.4f} {v-b:>+10.4f}\n")
            f.write(f"  Train time : {r.get('train_time', 'N/A')}\n")
            f.write(f"  Infer speed: {r.get('infer_speed', 0):.1f} samples/sec\n")
            f.write(f"  Model dir  : {r.get('model_dir', 'N/A')}\n\n")

        # Summary
        success = [r for r in results if r["status"] == "SUCCESS"]
        if success:
            best = max(success, key=lambda r: r.get("overall_f1", 0))
            f.write("=" * 100 + "\n")
            f.write(f"WINNER (Best Post-FT F1): {best['name']}\n")
            f.write(f"  F1     : {best['overall_f1']:.4f}  (Δ {best.get('delta_f1', 0):+.4f})\n")
            f.write(f"  nDCG@{ndcg_k}: {best.get(f'ndcg_at_{ndcg_k}', 0):.4f}  "
                    f"(Δ {best.get(f'delta_ndcg_at_{ndcg_k}', 0):+.4f})\n")

    print(f"[Benchmark GLiNER] TXT: {txt_path}")
    return csv_path, txt_path


# ================================================================
# Phân tích phân bố entity trong dataset
# ================================================================
def analyze_entity_distribution(samples: List[Dict], entity_types: List[str]) -> Dict:
    """
    Phân tích phân bố entity trong dataset.
    Giúp hiểu data để điều chỉnh config (neg_type_ratio, max_length, ...).

    In ra:
    - Số entity per type, span length distribution
    - Text length distribution và khuyến nghị max_length
    - Tỉ lệ bị cắt theo các max_length khác nhau

    Args:
        samples     : List {text, entities}
        entity_types: Danh sách entity type

    Returns:
        Dict thống kê
    """
    import statistics
    from collections import Counter

    print_banner("PHÂN TÍCH PHÂN BỐ ENTITY (DATA ANALYSIS)")

    type_counts          = Counter()
    type_span_lens       = defaultdict(list)
    text_lens            = []
    entity_counts_per_s  = []

    for s in samples:
        text_lens.append(len(s["text"]))
        entity_counts_per_s.append(len(s["entities"]))
        for ent in s["entities"]:
            lbl = ent["label"].upper()
            type_counts[lbl] += 1
            type_span_lens[lbl].append(ent["end"] - ent["start"])

    total_ents = sum(type_counts.values())
    print(f"Tổng số sample: {len(samples)}")
    print(f"\nPhân bố entity type:")
    for et in entity_types:
        et_up = et.upper()
        cnt   = type_counts.get(et_up, 0)
        spans = type_span_lens.get(et_up, [])
        avg_s = statistics.mean(spans) if spans else 0
        med_s = statistics.median(spans) if spans else 0
        p90_s = sorted(spans)[int(len(spans) * 0.9)] if spans else 0
        print(f"  {et_up:<14}: {cnt:>6} ({cnt/max(total_ents,1)*100:.1f}%) | "
              f"span avg={avg_s:.1f} med={med_s:.1f} p90={p90_s:.1f} chars")

    avg_ent = statistics.mean(entity_counts_per_s)
    print(f"\nEntity per sample: avg={avg_ent:.1f}, max={max(entity_counts_per_s, default=0)}")
    print(f"\nText length (chars):")
    print(f"  mean={statistics.mean(text_lens):.0f}  "
          f"median={statistics.median(text_lens):.0f}  "
          f"max={max(text_lens, default=0)}  "
          f"p90={sorted(text_lens)[int(len(text_lens)*0.9)]:.0f}")

    print(f"\nTỉ lệ bị cắt (~4 chars/token):")
    for lim in [512, 768, 1024, 2048]:
        cut = sum(1 for l in text_lens if l > lim * 4) / max(len(text_lens), 1)
        print(f"  max_length={lim:>4}: {cut*100:.1f}% bị cắt")

    print(f"\n→ Khuyến nghị max_length  : 1024 (bắt >80% text đầy đủ)")
    print(f"→ Khuyến nghị neg_type_ratio: {len(entity_types)} (1 per entity type)")

    return {
        "type_counts": dict(type_counts),
        "text_len_mean": statistics.mean(text_lens),
        "text_len_max":  max(text_lens, default=0),
        "entity_per_sample_mean": avg_ent,
    }


# ================================================================
# Điều phối benchmark toàn bộ
# ================================================================
def run_all_gliner_benchmarks(cfg: dict) -> List[Dict[str, Any]]:
    """
    Điều phối benchmark toàn bộ GLiNER variants.

    Luồng:
    1. Load dataset (1 lần)
    2. Chuyển sang GLiNER format
    3. Data analysis (nếu bật)
    4. Lần lượt: Baseline eval → Fine-tune → Post-FT eval cho từng model
    5. In bảng so sánh (Baseline vs Post-FT, per F1 và nDCG@10)
    6. Lưu CSV + TXT
    7. Chạy full pipeline analysis trên model tốt nhất

    Args:
        cfg: Config dict từ config.yaml

    Returns:
        List kết quả benchmark
    """
    print_banner("BENCHMARK GLiNER ARCHITECTURES (Baseline → FT → Post-FT)")

    bench_cfg = cfg.get("benchmark_gliner", {})
    data_cfg  = cfg["data"]
    seed      = cfg["run"].get("seed", 42)
    ndcg_k    = bench_cfg.get("ndcg_k", 10)

    set_seed(seed)

    entity_types = bench_cfg.get(
        "entity_types",
        cfg.get("gliner", {}).get("entity_types", ["SKILL", "EXPERIENCE"])
    )
    entity_types = [et.upper() for et in entity_types]
    print(f"[Benchmark GLiNER] Entity types : {entity_types}")
    print(f"[Benchmark GLiNER] Metrics      : F1 + nDCG@{ndcg_k}")

    # 1. Load dataset
    print("[Benchmark GLiNER] Load dataset...")
    train_data, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.1),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
    )

    # 2. Chuyển sang GLiNER format
    max_length = bench_cfg.get("max_length", 1024)
    train_samples = prepare_gliner_samples(train_data, entity_types, max_length)
    val_samples   = prepare_gliner_samples(val_data,   entity_types, max_length)
    print(f"[Benchmark GLiNER] Train: {len(train_samples)} | Val: {len(val_samples)}")

    # 3. Data analysis
    if bench_cfg.get("analyze_data", True):
        analyze_entity_distribution(train_samples + val_samples, entity_types)

    # 4. Danh sách model
    model_list = bench_cfg.get("models", [])
    enabled    = [m for m in model_list if m.get("enabled", True)]
    if not enabled:
        print("[LỖI] Không có model nào enabled trong benchmark_gliner.models!")
        return []

    print(f"\n[Benchmark GLiNER] {len(enabled)}/{len(model_list)} model sẽ chạy:")
    for m in enabled:
        print(f"  - {m.get('name', m['model_name'])} ({m['model_name']})")
    print()

    # 5. Train + eval từng model
    all_results   = []
    overall_start = time.time()

    for i, model_cfg_item in enumerate(enabled, 1):
        name = model_cfg_item.get("name", model_cfg_item["model_name"])
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(enabled)}] {name}")
        print(f"{'='*70}")

        r = run_single_gliner_benchmark(
            model_cfg=model_cfg_item,
            benchmark_cfg=bench_cfg,
            train_samples=train_samples,
            val_samples=val_samples,
            entity_types=entity_types,
            seed=seed,
        )
        all_results.append(r)

        if r["status"] == "SUCCESS":
            print(
                f"✓ {name}: "
                f"F1={r['overall_f1']:.4f} (Δ{r.get('delta_f1',0):+.4f}) | "
                f"nDCG@{ndcg_k}={r.get(f'ndcg_at_{ndcg_k}', 0):.4f} | "
                f"Time={r['train_time']}"
            )
        else:
            print(f"✗ {name}: FAILED - {r.get('error', '')[:60]}")

    total_time = format_time(time.time() - overall_start)
    print(f"\n[Benchmark GLiNER] Tất cả xong. Tổng: {total_time}")

    # 6. In bảng + lưu
    print_gliner_results_table(all_results, entity_types, ndcg_k)
    output_dir = bench_cfg.get("output_dir", "./outputs/benchmark_gliner")
    save_gliner_results(all_results, entity_types, output_dir, ndcg_k)

    # 7. Full pipeline cho model tốt nhất
    success = [r for r in all_results if r["status"] == "SUCCESS"]
    if success and bench_cfg.get("run_best_model_pipeline", True):
        best = max(success, key=lambda r: r.get("overall_f1", 0))
        print(f"\n[Benchmark GLiNER] Chạy full pipeline cho model tốt nhất: {best['name']}")
        run_best_model_full_pipeline(
            best_result=best,
            benchmark_cfg=bench_cfg,
            train_samples=train_samples,
            val_samples=val_samples,
            entity_types=entity_types,
            seed=seed,
        )

    return all_results


# ================================================================
# Entry point độc lập
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark GLiNER Architectures: Baseline → Fine-tune → Post-FT (F1 + nDCG@10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python benchmark_gliner.py                         # Dùng config.yaml
  python benchmark_gliner.py --config my_cfg.yaml   # Config khác
        """
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    if not os.path.exists(config_path):
        alt = str(Path(__file__).parent / args.config)
        config_path = alt if os.path.exists(alt) else config_path

    cfg = load_config(config_path)

    # Resolve paths
    ds = cfg["data"]["dataset_path"]
    if not os.path.isabs(ds):
        cfg["data"]["dataset_path"] = str((Path(config_path).parent / ds).resolve())

    bench_out = cfg.get("benchmark_gliner", {}).get("output_dir", "./outputs/benchmark_gliner")
    if not os.path.isabs(bench_out):
        if "benchmark_gliner" not in cfg:
            cfg["benchmark_gliner"] = {}
        cfg["benchmark_gliner"]["output_dir"] = str(
            (Path(config_path).parent / bench_out).resolve()
        )

    run_all_gliner_benchmarks(cfg)
