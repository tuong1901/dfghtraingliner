# -*- coding: utf-8 -*-
"""
eval_master.py
--------------
Script điều phối đánh giá toàn diện sau khi huấn luyện.
1. Chạy đánh giá (Inference) trên tập Test độc lập cho cả GLiNER và Level Classifier.
2. In bảng chỉ số chi tiết (Precision, Recall, F1-Score, Accuracy).
3. Hiển thị bảng ma trận nhầm lẫn (Confusion Matrix) dạng ASCII.
4. Trực quan hóa và vẽ biểu đồ Loss (loss_history.json) và Heatmap của Confusion Matrix, lưu thành file ảnh PNG tại outputs/figures/.
5. Tạo tệp báo cáo markdown tổng hợp outputs/evaluation_report.md.

Sử dụng:
    python eval_master.py                      # Đánh giá cả hai model (theo config.yaml)
    python eval_master.py --gliner             # Chỉ đánh giá GLiNER
    python eval_master.py --classifier         # Chỉ đánh giá Level Classifier
    python eval_master.py --config config.yaml # Chọn config cụ thể
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Sử dụng Agg backend cho matplotlib để chạy không giao diện (headless) trên Kaggle/Colab/Server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Cấu hình sys.path để import các module cục bộ
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, load_dataset, set_seed, check_device, print_banner
from train_gliner import prepare_gliner_samples
from eval_gliner import compute_ner_confusion_matrix, print_confusion_matrix as print_ner_cm, compute_metrics_from_cm

# Màu sắc đồ họa chủ đạo
BLUE, GREEN, ORANGE, RED, PURPLE = "#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"

def setup_plot_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    })

def load_training_history(output_dir: str) -> List[Dict]:
    """
    Tải lịch sử training từ loss_history.json. Nếu không tìm thấy,
    tự động quét và khôi phục từ tệp trainer_state.json của các checkpoint.
    """
    history_path = os.path.join(output_dir, "loss_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    # Thử khôi phục từ trainer_state.json trong các thư mục checkpoint-*
    import glob
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoint_dirs:
        # Sắp xếp theo số bước giảm dần để lấy checkpoint mới nhất
        def get_step_num(d):
            try:
                return int(os.path.basename(d).split("-")[-1])
            except ValueError:
                return -1
        checkpoint_dirs.sort(key=get_step_num, reverse=True)
        
        for check_dir in checkpoint_dirs:
            state_path = os.path.join(check_dir, "trainer_state.json")
            if os.path.exists(state_path):
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                        if "log_history" in state_data:
                            print(f"[eval_master] Tự động khôi phục lịch sử training từ checkpoint: {state_path}")
                            return state_data["log_history"]
                except Exception:
                    pass
    return []

def plot_loss_curve(history_data: List[Dict], title: str, save_path: str, is_classifier: bool = False):
    """
    Vẽ biểu đồ Loss curve từ dữ liệu lịch sử training.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    if is_classifier:
        # Lọc dữ liệu từ classifier history
        train_steps = []
        train_losses = []
        val_steps = []
        val_losses = []
        
        for item in history_data:
            step = item.get("step")
            if "loss" in item:
                train_steps.append(step)
                train_losses.append(item["loss"])
            elif "eval_loss" in item:
                val_steps.append(step)
                val_losses.append(item["eval_loss"])
                
        if train_losses:
            ax.plot(train_steps, train_losses, label="Train Loss", color=BLUE, linewidth=1.5)
        if val_losses:
            ax.plot(val_steps, val_losses, label="Validation Loss", color=ORANGE, marker="o", markersize=4, linestyle="--")
    else:
        # GLiNER Trainer log format
        steps = []
        train_losses = []
        val_steps = []
        val_losses = []
        
        for item in history_data:
            step = item.get("step")
            if "loss" in item:
                steps.append(step)
                train_losses.append(item["loss"])
            elif "eval_loss" in item:
                val_steps.append(step)
                val_losses.append(item["eval_loss"])
                
        if train_losses:
            ax.plot(steps, train_losses, label="Train Loss", color=BLUE, linewidth=1.5)
        if val_losses:
            ax.plot(val_steps, val_losses, label="Validation Loss", color=ORANGE, marker="o", markersize=4, linestyle="--")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Đã vẽ biểu đồ Loss tại: {save_path}")

def plot_confusion_matrix_heatmap(cm_matrix: np.ndarray, labels: List[str], title: str, save_path: str, color_theme: str = "blue"):
    """
    Vẽ biểu đồ Heatmap cho ma trận nhầm lẫn Confusion Matrix.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cm_matrix = np.array(cm_matrix, dtype=float)
    
    # Định nghĩa color map
    color_hex = BLUE if color_theme == "blue" else GREEN
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ffffff", color_hex])
    
    im = ax.imshow(cm_matrix, cmap=cmap, aspect="auto")
    
    # Xoay label trục X
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True / Gold Label", fontweight="bold")
    ax.set_title(title, pad=15)
    
    # Điền giá trị text vào từng ô
    thresh = cm_matrix.max() * 0.6
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            val = int(cm_matrix[i, j])
            ax.text(j, i, f"{val:,}" if val > 0 else "0", 
                    ha="center", va="center",
                    color="white" if cm_matrix[i, j] > thresh else "black", 
                    fontsize=8.5)
            
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[✓] Đã vẽ biểu đồ Confusion Matrix Heatmap tại: {save_path}")


# ===========================================================================
# 1. Đánh giá GLiNER NER Model
# ===========================================================================
def run_gliner_eval(cfg: dict, figure_dir: str) -> Dict[str, Any]:
    print_banner("ĐÁNH GIÁ MÔ HÌNH GLiNER NER")
    
    gcfg = cfg["gliner"]
    output_dir = gcfg.get("output_dir", "./outputs/gliner")
    model_dir = os.path.join(output_dir, "final_model")
    
    if not os.path.exists(model_dir):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục mô hình GLiNER tại: {model_dir}")
        print("Bỏ qua đánh giá GLiNER. Cần chạy train mô hình trước.")
        return {}
        
    entity_types = gcfg.get("entity_types", ["SKILL", "EXPERIENCE"])
    entity_types = [et.upper() for et in entity_types]
    
    # Tải dataset (lấy đúng tập Test 10% độc lập giống lúc train)
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)
    set_seed(seed)
    
    print("[GLiNER] Đang nạp dataset...")
    _, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.2),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
    )
    
    # Split Val và Test (50/50)
    import random
    random.seed(seed)
    random.shuffle(val_data)
    split_idx = len(val_data) // 2
    test_data = val_data[:split_idx]
    
    max_length = gcfg.get("max_length", 512)
    print(f"[GLiNER] Chuẩn bị {len(test_data)} mẫu test (filter_empty=False)...")
    test_samples = prepare_gliner_samples(test_data, entity_types, max_length, filter_empty=False)
    
    # Load model
    device = check_device()
    print(f"[GLiNER] Đang tải mô hình từ {model_dir} lên {device}...")
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained(model_dir)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[LỖI] Không thể nạp mô hình GLiNER: {e}")
        return {}

    # Chạy inference
    print("[GLiNER] Đang dự đoán trên tập TEST...")
    texts = [s["text"] for s in test_samples]
    gold_entities_list = [s["entities"] for s in test_samples]
    
    all_predictions = []
    batch_size = gcfg.get("eval_batch_size", 8)
    
    start_time = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_preds = model.batch_predict_entities(batch_texts, entity_types, threshold=0.5)
        except AttributeError:
            batch_preds = [
                model.predict_entities(t, entity_types, threshold=0.5)
                for t in batch_texts
            ]
        all_predictions.extend(batch_preds)
        
    elapsed = time.time() - start_time
    print(f"[GLiNER] Inference hoàn tất sau {elapsed:.2f}s ({len(texts)/elapsed:.2f} samples/sec).")
    
    # Tính Confusion Matrix
    cm = compute_ner_confusion_matrix(all_predictions, gold_entities_list, entity_types)
    
    # In ra terminal
    labels_cm = entity_types + ["O"]
    print_ner_cm(cm, labels_cm)
    metrics = compute_metrics_from_cm(cm, entity_types)
    
    # Tính nDCG@5 và nDCG@10
    try:
        from benchmark_gliner import compute_ndcg_corpus
        ndcg_5 = compute_ndcg_corpus(all_predictions, gold_entities_list, k=5)
        ndcg_10 = compute_ndcg_corpus(all_predictions, gold_entities_list, k=10)
        print("="*28 + " ĐÁNH GIÁ ĐỘ PHÙ HỢP XẾP HẠNG (nDCG) " + "="*28)
        print(f"  nDCG@5  : {ndcg_5:.4f}")
        print(f"  nDCG@10 : {ndcg_10:.4f}")
        print("="*84 + "\n")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể tính toán nDCG: {e}")
        ndcg_5, ndcg_10 = 0.0, 0.0
    
    # Chuyển cm dict thành np.ndarray để vẽ đồ thị
    cm_matrix = np.zeros((len(labels_cm), len(labels_cm)), dtype=int)
    for i, g_lbl in enumerate(labels_cm):
        for j, p_lbl in enumerate(labels_cm):
            cm_matrix[i, j] = cm[g_lbl][p_lbl]
            
    # Vẽ Heatmap
    heatmap_path = os.path.join(figure_dir, "gliner_confusion_matrix.png")
    plot_confusion_matrix_heatmap(cm_matrix, labels_cm, "GLiNER NER Confusion Matrix (Overlap Match)", heatmap_path, "blue")
    
    # Vẽ Loss Curve
    history = load_training_history(output_dir)
    loss_img_path = os.path.join(figure_dir, "gliner_loss_curve.png")
    loss_plotted = False
    if history:
        try:
            plot_loss_curve(history, "GLiNER Training & Validation Loss", loss_img_path, is_classifier=False)
            loss_plotted = True
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể vẽ Loss curve cho GLiNER từ lịch sử tìm được: {e}")
            
    # Tạo cấu trúc kết quả trả về
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "heatmap_img": "gliner_confusion_matrix.png",
        "loss_img": "gliner_loss_curve.png" if loss_plotted else None,
        "elapsed": elapsed,
        "ndcg_at_5": ndcg_5,
        "ndcg_at_10": ndcg_10
    }


# ===========================================================================
# 2. Đánh giá Level Classifier Model
# ===========================================================================
def run_classifier_eval(cfg: dict, figure_dir: str) -> Dict[str, Any]:
    print_banner("ĐÁNH GIÁ MÔ HÌNH LEVEL CLASSIFIER")
    
    ccfg = cfg["classifier"]
    output_dir = ccfg.get("output_dir", "./outputs/classifier")
    model_dir = os.path.join(output_dir, "best_model")
    
    if not os.path.exists(model_dir):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục mô hình Level Classifier tại: {model_dir}")
        print("Bỏ qua đánh giá Level Classifier. Cần chạy train mô hình trước.")
        return {}
        
    # Đọc cấu hình map nhãn
    label_map_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            lmap = json.load(f)
            level_labels = lmap["level_labels"]
    else:
        level_labels = ccfg.get("level_labels", ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"])
        
    num_labels = len(level_labels)
    
    # Tải dataset (lấy đúng tập Test 10% độc lập giống lúc train)
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)
    set_seed(seed)
    
    print("[Classifier] Đang nạp dataset...")
    _, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.2),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
        level_labels=level_labels,
    )
    
    # Split Val và Test (50/50)
    import random
    random.seed(seed)
    random.shuffle(val_data)
    split_idx = len(val_data) // 2
    test_data = val_data[:split_idx]
    
    # Load tokenizer và model
    device = check_device()
    print(f"[Classifier] Đang tải tokenizer và mô hình từ {model_dir} lên {device}...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from torch.utils.data import DataLoader
        from train_classifier import JobLevelDataset, collate_fn, evaluate
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
    except Exception as e:
        print(f"[LỖI] Không thể nạp mô hình Level Classifier: {e}")
        return {}
        
    max_length = ccfg.get("max_length", 512)
    truncation_strategy = ccfg.get("truncation_strategy", "head+tail")
    eval_batch_size = ccfg.get("eval_batch_size", 32)
    
    test_dataset = JobLevelDataset(
        test_data, tokenizer, level_labels, max_length, truncation_strategy
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print("[Classifier] Đang dự đoán trên tập TEST...")
    start_time = time.time()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
                
            outputs = model(**kwargs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    elapsed = time.time() - start_time
    print(f"[Classifier] Inference hoàn tất sau {elapsed:.2f}s.")
    
    # Tính các metrics
    from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    present_labels = sorted(list(set(all_labels + all_preds)))
    present_names = [level_labels[i] for i in present_labels]
    
    report_dict = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    )
    
    print("\n[Classifier] Classification Report:")
    print(report_text)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    print("[Classifier] Confusion Matrix:")
    max_len = max(len(name) for name in present_names)
    max_len = max(max_len, 11)
    
    header = f"{'True \\ Pred':<{max_len}} |" + "".join(f" {present_names[i]:<{max_len}}" for i in range(len(present_labels)))
    print(header)
    print("-" * len(header))
    for i in range(len(present_labels)):
        row_str = f"{present_names[i]:<{max_len}} |" + "".join(f" {cm[i, j]:<{max_len}}" for j in range(len(present_labels)))
        print(row_str)
    print()
    
    # Vẽ Heatmap
    heatmap_path = os.path.join(figure_dir, "classifier_confusion_matrix.png")
    plot_confusion_matrix_heatmap(cm, present_names, "Level Classifier Confusion Matrix (DistilBERT)", heatmap_path, "green")
    
    # Vẽ Loss Curve
    history = load_training_history(output_dir)
    loss_img_path = os.path.join(figure_dir, "classifier_loss_curve.png")
    loss_plotted = False
    if history:
        try:
            plot_loss_curve(history, "Classifier Training & Validation Loss", loss_img_path, is_classifier=True)
            loss_plotted = True
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể vẽ Loss curve cho Classifier từ lịch sử tìm được: {e}")
            
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "present_names": present_names,
        "heatmap_img": "classifier_confusion_matrix.png",
        "loss_img": "classifier_loss_curve.png" if loss_plotted else None,
        "elapsed": elapsed
    }


# ===========================================================================
# 3. Tạo báo cáo markdown tổng hợp
# ===========================================================================
def write_md_report(gliner_res: dict, clf_res: dict, report_path: str):
    """
    Tạo tệp markdown tổng kết kết quả đẹp mắt.
    """
    lines = []
    lines.append("# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH")
    lines.append(f"*(Tự động tạo lúc: {time.strftime('%Y-%m-%d %H:%M:%S')} - Test Set 10% độc lập)*\n")
    
    lines.append("## 1. Mô hình GLiNER NER (Trích xuất SKILL, EXPERIENCE)")
    if gliner_res:
        lines.append("| Nhãn thực thể | Precision | Recall | F1-Score |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for etype, score in gliner_res["metrics"].items():
            if etype != "OVERALL":
                lines.append(f"| **{etype}** | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} |")
        
        overall = gliner_res["metrics"].get("OVERALL", {"precision":0, "recall":0, "f1":0})
        lines.append(f"| **TỔNG THỂ (OVERALL)** | **{overall['precision']:.4f}** | **{overall['recall']:.4f}** | **{overall['f1']:.4f}** |")
        
        # Thêm nDCG nếu có
        if "ndcg_at_5" in gliner_res:
            lines.append(f"\n- **nDCG@5**: `{gliner_res['ndcg_at_5']:.4f}`")
        if "ndcg_at_10" in gliner_res:
            lines.append(f"- **nDCG@10**: `{gliner_res['ndcg_at_10']:.4f}`")
            
        lines.append(f"\n*Thời gian dự đoán tập Test: {gliner_res['elapsed']:.2f} giây.*")
        
        # Nhúng hình
        lines.append("\n### Trực quan hóa kết quả GLiNER:")
        lines.append("#### A. Ma trận nhầm lẫn (Confusion Matrix):")
        lines.append(f"![GLiNER Confusion Matrix](figures/{gliner_res['heatmap_img']})")
        if gliner_res.get("loss_img"):
            lines.append("\n#### B. Đường cong học tập (Loss Curve):")
            lines.append(f"![GLiNER Loss Curve](figures/{gliner_res['loss_img']})")
    else:
        lines.append("*Chưa có kết quả hoặc mô hình chưa được huấn luyện.*\n")
        
    lines.append("\n" + "-"*50 + "\n")
    
    lines.append("## 2. Mô hình Level Classifier (Phân loại Cấp bậc công việc)")
    if clf_res:
        lines.append(f"- **Độ chính xác tổng thể (Accuracy)**: `{clf_res['accuracy'] * 100:.2f}%`")
        lines.append(f"- **F1-Score (Weighted)**: `{clf_res['f1_weighted']:.4f}`")
        lines.append(f"- **F1-Score (Macro)**: `{clf_res['f1_macro']:.4f}`\n")
        
        lines.append("| Cấp bậc | Precision | Recall | F1-Score | Support |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for name in clf_res["present_names"]:
            metric = clf_res["report_dict"][name]
            lines.append(f"| **{name}** | {metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1-score']:.4f} | {int(metric['support'])} |")
            
        lines.append(f"\n*Thời gian dự đoán tập Test: {clf_res['elapsed']:.2f} giây.*")
        
        # Nhúng hình
        lines.append("\n### Trực quan hóa kết quả Level Classifier:")
        lines.append("#### A. Ma trận nhầm lẫn (Confusion Matrix):")
        lines.append(f"![Classifier Confusion Matrix](figures/{clf_res['heatmap_img']})")
        if clf_res.get("loss_img"):
            lines.append("\n#### B. Đường cong học tập (Loss Curve):")
            lines.append(f"![Classifier Loss Curve](figures/{clf_res['loss_img']})")
    else:
        lines.append("*Chưa có kết quả hoặc mô hình chưa được huấn luyện.*\n")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"\n[✓] Đã tạo báo cáo markdown tổng hợp tại: {report_path}")


# ===========================================================================
# 4. Entry Point
# ===========================================================================
if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="Master Evaluation Script - GLiNER NER + Level Classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Đường dẫn tới file config.yaml")
    parser.add_argument("--gliner", action="store_true", help="Chỉ đánh giá mô hình GLiNER NER")
    parser.add_argument("--classifier", action="store_true", help="Chỉ đánh giá mô hình Level Classifier")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Resolve relative paths
    base_dir = Path(__file__).resolve().parent
    ds_path = cfg["data"]["dataset_path"]
    if not os.path.isabs(ds_path):
        cfg["data"]["dataset_path"] = str((base_dir / ds_path).resolve())
        
    for model_key in ["gliner", "classifier"]:
        if model_key in cfg and "output_dir" in cfg[model_key]:
            out = cfg[model_key]["output_dir"]
            if not os.path.isabs(out):
                cfg[model_key]["output_dir"] = str((base_dir / out).resolve())

    # Thư mục chứa hình ảnh đầu ra
    figure_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(figure_dir, exist_ok=True)
    
    setup_plot_style()
    
    run_all = not args.gliner and not args.classifier
    
    gliner_results = {}
    classifier_results = {}
    
    if args.gliner or run_all:
        try:
            gliner_results = run_gliner_eval(cfg, figure_dir)
        except Exception as e:
            print(f"[CẢNH BÁO] Đánh giá GLiNER thất bại: {e}")
            import traceback
            traceback.print_exc()
            
    if args.classifier or run_all:
        try:
            # import torch inside because checking device might be delayed
            import torch
            classifier_results = run_classifier_eval(cfg, figure_dir)
        except Exception as e:
            print(f"[CẢNH BÁO] Đánh giá Level Classifier thất bại: {e}")
            import traceback
            traceback.print_exc()
            
    # Tạo báo cáo tổng hợp
    report_path = os.path.join(base_dir, "outputs", "evaluation_report.md")
    write_md_report(gliner_results, classifier_results, report_path)
    
    print("\n" + "="*60)
    print("  ĐÁNH GIÁ VÀ TRỰC QUAN HÓA TOÀN DIỆN ĐÃ HOÀN THÀNH!")
    print("="*60)
    print(f"  - Các biểu đồ đã lưu tại: {figure_dir}/")
    print(f"  - File báo cáo tổng hợp: {report_path}")
    print("="*60 + "\n")
