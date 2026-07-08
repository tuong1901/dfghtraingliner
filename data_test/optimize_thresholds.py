import os
import sys
import json
import pandas as pd
from collections import defaultdict

# Force UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def overlaps(s1, s2):
    return s1[2] == s2[2] and max(s1[0], s2[0]) < min(s1[1], s2[1])

def calculate_metrics(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def evaluate_thresholds(gold_data, pred_map, df_pred, threshold_skill, threshold_exp):
    entity_types = ["SKILL", "EXPERIENCE"]
    
    # Overlap metrics
    overlap_tp = defaultdict(int)
    overlap_fp = defaultdict(int)
    overlap_fn = defaultdict(int)
    
    # Exact metrics
    exact_tp = defaultdict(int)
    exact_fp = defaultdict(int)
    exact_fn = defaultdict(int)

    for idx, gold_item in enumerate(gold_data):
        gold_text = gold_item.get("text", "")
        norm_gold = "".join(gold_text.lower().split())

        pred_entities = []
        if norm_gold in pred_map:
            pred_entities = pred_map[norm_gold]
        elif idx < len(df_pred):
            pred_row = df_pred.iloc[idx]
            pred_raw_json = pred_row.get("predicted_entities_raw_json")
            try:
                pred_entities = json.loads(pred_raw_json) if isinstance(pred_raw_json, str) else []
            except Exception:
                pred_entities = []

        gold_spans_list = [
            (span[0], span[1], span[2].upper())
            for span in gold_item.get("label", [])
            if span[2].upper() in entity_types
        ]
        gold_spans_set = set(gold_spans_list)

        gold_major_spans = [
            (span[0], span[1])
            for span in gold_item.get("label", [])
            if span[2].upper() == "MAJOR"
        ]

        pred_spans_list = []
        for ent in pred_entities:
            lbl = ent.get("label", "").upper()
            score = ent.get("score", 1.0)
            
            if lbl == "SKILL" and score < threshold_skill:
                continue
            if lbl == "EXPERIENCE" and score < threshold_exp:
                continue
                
            if lbl in entity_types:
                start = ent["start"]
                end = ent["end"]
                
                if lbl == "SKILL":
                    is_major = False
                    for m_start, m_end in gold_major_spans:
                        if max(start, m_start) < min(end, m_end):
                            is_major = True
                            break
                    if is_major:
                        continue
                pred_spans_list.append((start, end, lbl))

        pred_spans_set = set(pred_spans_list)

        # Exact Match TP/FP/FN
        for span in pred_spans_set:
            lbl = span[2]
            if span in gold_spans_set:
                exact_tp[lbl] += 1
            else:
                exact_fp[lbl] += 1
        for span in gold_spans_set:
            lbl = span[2]
            if span not in pred_spans_set:
                exact_fn[lbl] += 1

        # Overlap Match TP/FP/FN
        matched_preds = set()
        matched_golds = set()

        for p_idx, p in enumerate(pred_spans_list):
            for g_idx, g in enumerate(gold_spans_list):
                if overlaps(p, g):
                    matched_preds.add(p_idx)
                    matched_golds.add(g_idx)

        for p_idx, p in enumerate(pred_spans_list):
            lbl = p[2]
            if p_idx in matched_preds:
                overlap_tp[lbl] += 1
            else:
                overlap_fp[lbl] += 1

        for g_idx, g in enumerate(gold_spans_list):
            lbl = g[2]
            if g_idx not in matched_golds:
                overlap_fn[lbl] += 1

    total_exact_tp = sum(exact_tp.values())
    total_exact_fp = sum(exact_fp.values())
    total_exact_fn = sum(exact_fn.values())
    _, _, exact_f1 = calculate_metrics(total_exact_tp, total_exact_fp, total_exact_fn)

    total_overlap_tp = sum(overlap_tp.values())
    total_overlap_fp = sum(overlap_fp.values())
    total_overlap_fn = sum(overlap_fn.values())
    
    skill_prec, skill_rec, skill_f1 = calculate_metrics(overlap_tp["SKILL"], overlap_fp["SKILL"], overlap_fn["SKILL"])
    exp_prec, exp_rec, exp_f1 = calculate_metrics(overlap_tp["EXPERIENCE"], overlap_fp["EXPERIENCE"], overlap_fn["EXPERIENCE"])
    overlap_prec, overlap_rec, overlap_f1 = calculate_metrics(total_overlap_tp, total_overlap_fp, total_overlap_fn)

    return {
        "exact_f1": exact_f1,
        "overlap_prec": overlap_prec,
        "overlap_rec": overlap_rec,
        "overlap_f1": overlap_f1,
        "skill_f1": skill_f1,
        "skill_prec": skill_prec,
        "skill_rec": skill_rec,
        "exp_f1": exp_f1,
        "exp_prec": exp_prec,
        "exp_rec": exp_rec,
        "total_preds": total_overlap_tp + total_overlap_fp
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gold_json_path = os.path.join(base_dir, "data_xin_1000_dong_gold.json")
    predicted_excel_path = os.path.join(base_dir, "data_xin_1000_dong_predicted.xlsx")

    if not os.path.exists(gold_json_path) or not os.path.exists(predicted_excel_path):
        print("❌ Required files not found!")
        sys.exit(1)

    with open(gold_json_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    df_pred = pd.read_excel(predicted_excel_path)
    
    pred_map = {}
    text_col = "combined_text" if "combined_text" in df_pred.columns else df_pred.columns[0]
    for _, row in df_pred.iterrows():
        txt = str(row[text_col])
        norm_txt = "".join(txt.lower().split())
        pred_raw = row.get("predicted_entities_raw_json")
        try:
            pred_entities = json.loads(pred_raw) if isinstance(pred_raw, str) else []
        except Exception:
            pred_entities = []
        pred_map[norm_txt] = pred_entities

    print("Running Grid Search to optimize thresholds...")
    print(f"{'SKILL Thresh':<12} | {'EXP Thresh':<10} | {'Overlap F1':<10} | {'SKILL F1':<10} | {'EXP F1':<10} | {'Exact F1':<10} | {'Num Preds':<10}")
    print("-" * 85)

    best_overlap_f1 = 0.0
    best_results = None
    best_thresholds = (0, 0)

    skill_thresholds = [round(x * 0.01, 2) for x in range(75, 99)]
    exp_thresholds = [round(x * 0.05, 2) for x in range(2, 19)]

    for t_skill in skill_thresholds:
        for t_exp in exp_thresholds:
            res = evaluate_thresholds(gold_data, pred_map, df_pred, t_skill, t_exp)
            print(f"{t_skill:<12.2f} | {t_exp:<10.2f} | {res['overlap_f1']:<10.4f} | {res['skill_f1']:<10.4f} | {res['exp_f1']:<10.4f} | {res['exact_f1']:<10.4f} | {res['total_preds']:<10}")
            
            if res["overlap_f1"] > best_overlap_f1:
                best_overlap_f1 = res["overlap_f1"]
                best_results = res
                best_thresholds = (t_skill, t_exp)

    print("=" * 85)
    print("🏆 BEST CONFIGURATION FOUND:")
    print(f"  - SKILL Threshold       : {best_thresholds[0]:.2f}")
    print(f"  - EXPERIENCE Threshold  : {best_thresholds[1]:.2f}")
    print(f"  - Overall Overlap F1    : {best_results['overlap_f1']*100:.2f}% (P={best_results['overlap_prec']*100:.2f}%, R={best_results['overlap_rec']*100:.2f}%)")
    print(f"  - SKILL Overlap F1      : {best_results['skill_f1']*100:.2f}% (P={best_results['skill_prec']*100:.2f}%, R={best_results['skill_rec']*100:.2f}%)")
    print(f"  - EXPERIENCE Overlap F1 : {best_results['exp_f1']*100:.2f}% (P={best_results['exp_prec']*100:.2f}%, R={best_results['exp_rec']*100:.2f}%)")
    print(f"  - Overall Exact F1      : {best_results['exact_f1']*100:.2f}%")
    print(f"  - Number of Predictions : {best_results['total_preds']}")

if __name__ == "__main__":
    main()
