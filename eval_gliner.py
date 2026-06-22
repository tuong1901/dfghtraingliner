"""
eval_gliner.py
--------------
Script đánh giá chi tiết mô hình GLiNER đã huấn luyện trên tập dữ liệu Test độc lập.
Tính toán Precision, Recall, F1 và hiển thị bảng Confusion Matrix (NER Span-level).

Sử dụng:
    python eval_gliner.py --model_dir ./outputs/gliner_small_v25_major/final_model --config config_small_v25_major.yaml
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, load_dataset, set_seed, check_device, print_banner
from train_gliner import prepare_gliner_samples


# ================================================================
# Tính Confusion Matrix cho bài toán NER
# ================================================================
def compute_ner_confusion_matrix(predictions: List[List[Dict]], gold_labels: List[List[Dict]], entity_types: List[str]):
    """
    Tính ma trận nhầm lẫn Confusion Matrix cho NER ở mức độ so khớp vị trí span.
    
    Các trường hợp:
    - Gold span khớp vị trí và khớp nhãn với Pred span -> Đúng nhãn (TP)
    - Gold span khớp vị trí nhưng sai nhãn với Pred span -> Nhầm nhãn
    - Gold span không được model đoán ra -> Bỏ sót (FN / nhãn thực tế là L nhưng đoán là O)
    - Pred span thừa không khớp với Gold span nào -> Đoán thừa (FP / nhãn thực tế là O nhưng đoán là L)
    """
    labels = entity_types + ["O"]
    cm = {g_lbl: {p_lbl: 0 for p_lbl in labels} for g_lbl in labels}
    
    for preds, golds in zip(predictions, gold_labels):
        g_spans = [(g["start"], g["end"], g["label"].upper()) for g in golds]
        p_spans = [(p["start"], p["end"], p["label"].upper()) for p in preds]
        
        matched_preds = set()
        matched_golds = set()
        
        # 1. So khớp từ Gold sang Pred
        for g_idx, g in enumerate(g_spans):
            g_start, g_end, g_lbl = g
            
            matched_p_idx = None
            # Tìm Pred span có vị trí overlap (giao nhau) với Gold span
            for p_idx, p in enumerate(p_spans):
                if p_idx in matched_preds:
                    continue
                p_start, p_end, p_lbl = p
                # Kiểm tra giao nhau ít nhất 1 ký tự
                if max(g_start, p_start) < min(g_end, p_end):
                    matched_p_idx = p_idx
                    break
            
            if matched_p_idx is not None:
                p_lbl = p_spans[matched_p_idx][2]
                cm[g_lbl][p_lbl] += 1
                matched_preds.add(matched_p_idx)
                matched_golds.add(g_idx)
            else:
                # Gold có thực thể nhưng model không tìm ra -> Bỏ sót (FN)
                cm[g_lbl]["O"] += 1
                
        # 2. Những Pred span còn lại không khớp với Gold nào -> Đoán thừa (FP)
        for p_idx, p in enumerate(p_spans):
            if p_idx not in matched_preds:
                p_lbl = p[2]
                cm["O"][p_lbl] += 1
                
    return cm


# ================================================================
# In Confusion Matrix dạng ASCII Table
# ================================================================
def print_confusion_matrix(cm: Dict, labels: List[str]):
    """In Confusion Matrix đẹp mắt dạng ASCII."""
    print("\n" + "="*25 + " CONFUSION MATRIX (NER SPAN OVERLAP MATCH) " + "="*25)
    header = f"{'Gold \\ Pred':<15} |"
    for lbl in labels:
        header += f" {lbl:>12}"
    print(header)
    print("-" * len(header))
    
    for g_lbl in labels:
        row_str = f"{g_lbl:<15} |"
        for p_lbl in labels:
            val = cm[g_lbl][p_lbl]
            row_str += f" {val:>12,}"
        print(row_str)
    print("=" * len(header))
    print("[*] Ghi chú:")
    print("  - Dòng cuối 'O': Số lượng thực thể mô hình đoán THỪA (False Positives).")
    print("  - Cột cuối 'O': Số lượng thực thể mô hình BỎ SÓT (False Negatives).")
    print("  - Đường chéo chính: Dự đoán đúng (True Positives).")
    print("  - Các ô khác: Dự đoán nhầm lẫn nhãn thực thể này sang nhãn thực thể kia.\n")


# ================================================================
# Tính Precision, Recall, F1-Score
# ================================================================
def compute_metrics_from_cm(cm: Dict, entity_types: List[str]):
    """Tính các chỉ số đánh giá từ Confusion Matrix."""
    results = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print("="*28 + " CHI TIẾT ĐÁNH GIÁ THỰC THỂ " + "="*28)
    print(f"{'Entity Type':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    print("-"*84)
    
    for etype in entity_types:
        tp = cm[etype][etype]
        # FP = tổng cột của etype trừ TP (các phần tử đoán nhầm sang etype + đoán thừa từ O)
        fp = sum(cm[g_lbl][etype] for g_lbl in cm) - tp
        # FN = tổng hàng của etype trừ TP (các phần tử bị đoán nhầm sang nhãn khác + bỏ sót sang O)
        fn = sum(cm[etype][p_lbl] for p_lbl in cm[etype]) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        print(f"{etype:<15} | {prec:>10.4f} | {rec:>10.4f} | {f1:>10.4f} | {tp:>6} | {fp:>6} | {fn:>6}")
        results[etype] = {"precision": prec, "recall": rec, "f1": f1}
        
    print("-"*84)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    
    print(f"{'OVERALL':<15} | {overall_p:>10.4f} | {overall_r:>10.4f} | {overall_f1:>10.4f} | {total_tp:>6} | {total_fp:>6} | {total_fn:>6}")
    print("="*84 + "\n")
    
    return results


# ================================================================
# Main Function
# ================================================================
def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="Đánh giá chi tiết mô hình GLiNER NER")
    parser.add_argument("--model_dir", type=str, required=True, help="Đường dẫn tới thư mục model đã train")
    parser.add_argument("--config", type=str, default="config.yaml", help="Đường dẫn tới file config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng tự tin confidence score khi lọc thực thể")
    args = parser.parse_args()
    
    # 1. Load config và kiểm tra model path
    cfg = load_config(args.config)
    model_dir = args.model_dir
    
    if not os.path.exists(model_dir):
        print(f"[LỖI] Thư mục mô hình không tồn tại: {model_dir}")
        sys.exit(1)
        
    print_banner(f"EVALUATING GLiNER MODEL: {os.path.basename(model_dir)}")
    print(f"[*] Cấu hình: {args.config} | Ngưỡng threshold: {args.threshold}")
    
    # 2. Đọc entity types từ model folder
    entity_types_file = os.path.join(model_dir, "entity_types.json")
    if os.path.exists(entity_types_file):
        with open(entity_types_file, "r", encoding="utf-8") as f:
            entity_types = json.load(f).get("entity_types", ["SKILL", "EXPERIENCE"])
    else:
        entity_types = cfg["gliner"].get("entity_types", ["SKILL", "EXPERIENCE"])
    
    entity_types = [et.upper() for et in entity_types]
    print(f"[*] Danh sách nhãn NER: {entity_types}")
    
    # 3. Load dataset và trích xuất Test set (đảm bảo đồng bộ với lúc train)
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)
    set_seed(seed)
    
    print("[*] Đang tải tập dữ liệu...")
    _, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.2),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
    )
    
    # Chia val_data thành Val và Test (50% / 50%) giống hệt lúc train
    import random
    random.seed(seed)
    random.shuffle(val_data)
    split_idx = len(val_data) // 2
    test_data = val_data[:split_idx]
    
    max_length = cfg["gliner"].get("max_length", 1024)
    print(f"[*] Tạo {len(test_data)} mẫu test format GLiNER (max_length={max_length})...")
    test_samples = prepare_gliner_samples(test_data, entity_types, max_length, filter_empty=False)
    
    # 4. Load model
    device = check_device()
    print(f"[*] Đang tải mô hình trên thiết bị: {device}...")
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained(model_dir)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[LỖI] Không thể nạp mô hình: {e}")
        sys.exit(1)
        
    # 5. Chạy dự đoán hàng loạt (Batch Inference)
    print("[*] Đang tiến hành dự đoán trên tập TEST...")
    texts = [s["text"] for s in test_samples]
    gold_entities_list = [s["entities"] for s in test_samples]
    
    all_predictions = []
    batch_size = cfg["gliner"].get("eval_batch_size", 8)
    
    start_time = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_preds = model.batch_predict_entities(batch_texts, entity_types, threshold=args.threshold)
        except AttributeError:
            batch_preds = [
                model.predict_entities(t, entity_types, threshold=args.threshold)
                for t in batch_texts
            ]
        all_predictions.extend(batch_preds)
        
    elapsed = time.time() - start_time
    print(f"[+] Dự đoán hoàn tất sau {elapsed:.2f}s ({len(texts)/elapsed:.2f} câu/giây).")
    
    # 6. Tính Confusion Matrix và Metrics
    cm = compute_ner_confusion_matrix(all_predictions, gold_entities_list, entity_types)
    print_confusion_matrix(cm, entity_types + ["O"])
    compute_metrics_from_cm(cm, entity_types)
    
    # Lưu báo cáo vào thư mục model
    report_path = os.path.join(model_dir, "test_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "entity_types": entity_types,
            "threshold": args.threshold,
            "confusion_matrix": cm,
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] Đã xuất báo cáo JSON chi tiết tại: {report_path}")


if __name__ == "__main__":
    main()
