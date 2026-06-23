# -*- coding: utf-8 -*-
"""
make_thesis_figures.py
======================
Sinh các hình trực quan hóa cho khóa luận (mục 3.2 / Results) từ số liệu thật
đã đo được. Số liệu được nhúng trực tiếp ở đầu file (lấy từ các báo cáo:
  - outputs/benmark_glinner.txt
  - outputs/benmark_bert.txt
  - data_test/benchmark_report.txt
). Sửa các dict bên dưới nếu bạn chạy lại và có số mới.

Chạy:
    python make_thesis_figures.py
Đầu ra: thư mục outputs/figures/*.png (300 DPI, nền trắng).

Chỉ phụ thuộc matplotlib + numpy:
    pip install matplotlib numpy
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # không cần màn hình
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# 0. Cấu hình chung
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})

BLUE, GREEN, ORANGE, RED, PURPLE = "#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [+] {path}")


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.005):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=8)


# ---------------------------------------------------------------------------
# 1. Benchmark các biến thể GLiNER (Post-FT): Overall F1 + nDCG@10
#    Nguồn: outputs/benmark_glinner.txt
# ---------------------------------------------------------------------------
def fig_gliner_benchmark():
    models = ["Small-v2.5", "Medium-v2.5", "Medium-v2.1", "Small-v2.1"]
    f1 =   [0.7754, 0.6795, 0.6784, 0.6657]
    ndcg = [0.9106, 0.8767, 0.8798, 0.8683]

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w/2, f1, w, label="Overall F1", color=BLUE)
    b2 = ax.bar(x + w/2, ndcg, w, label="nDCG@10", color=GREEN)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("GLiNER variants — Post-fine-tuning (in-distribution test)")
    ax.legend(loc="lower right")
    # đánh dấu model được chọn
    ax.annotate("Selected", xy=(0, 0.7754), xytext=(0.1, 0.55),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    _save(fig, "fig_gliner_benchmark.png")


# ---------------------------------------------------------------------------
# 2. Generalization gap: in-distribution vs independent (overlap match)
#    Nguồn: benmark_glinner.txt (in-dist) + data_test/benchmark_report.txt (independent)
# ---------------------------------------------------------------------------
def fig_generalization_gap():
    cats = ["SKILL", "EXPERIENCE", "Overall"]
    indist =      [0.8166, 0.9511, 0.8384]
    independent = [0.5792, 0.6992, 0.5904]

    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w/2, indist, w, label="In-distribution test", color=BLUE)
    b2 = ax.bar(x + w/2, independent, w, label="Independent real-world test", color=ORANGE)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("F1 (overlap match)"); ax.set_ylim(0, 1.05)
    ax.set_title("Generalization gap: in-distribution vs. real-world JDs")
    ax.legend(loc="lower left")
    _save(fig, "fig_generalization_gap.png")


# ---------------------------------------------------------------------------
# 3. Benchmark Level classifier (weighted F1)
#    Nguồn: outputs/benmark_bert.txt
# ---------------------------------------------------------------------------
def fig_classifier_benchmark():
    models = ["DistilBERT", "BERT-base", "RoBERTa-base", "ELECTRA-small"]
    f1 = [0.7633, 0.7631, 0.7405, 0.6226]
    colors = [GREEN if m == "DistilBERT" else BLUE for m in models]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(models, f1, color=colors, width=0.6)
    _bar_labels(ax, bars)
    ax.set_ylabel("Weighted F1"); ax.set_ylim(0, 0.9)
    ax.set_title("Level-classifier benchmark (selected: DistilBERT)")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "fig_classifier_benchmark.png")


# ---------------------------------------------------------------------------
# 4. Per-class P/R/F1 của DistilBERT
#    Nguồn: outputs/benmark_bert.txt
# ---------------------------------------------------------------------------
def fig_classifier_per_class():
    levels = ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"]
    P = [0.32, 0.78, 0.87, 0.79, 0.68, 0.70]
    R = [0.69, 0.68, 0.87, 0.72, 0.75, 0.66]
    F = [0.44, 0.73, 0.87, 0.75, 0.71, 0.68]

    x = np.arange(len(levels)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - w, P, w, label="Precision", color=BLUE)
    ax.bar(x,     R, w, label="Recall", color=GREEN)
    ax.bar(x + w, F, w, label="F1", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(levels, rotation=15)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("DistilBERT level classifier — per-class metrics")
    ax.legend(loc="lower right", ncol=3)
    ax.annotate("INTERN underperforms\n(low support, n=29)", xy=(0, 0.32),
                xytext=(0.6, 0.12), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    _save(fig, "fig_classifier_per_class.png")


# ---------------------------------------------------------------------------
# Helper: vẽ heatmap confusion matrix
# ---------------------------------------------------------------------------
def _heatmap(ax, cm, row_labels, col_labels, title, cmap):
    cm = np.array(cm, dtype=float)
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold / True")
    ax.set_title(title)
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.grid(False)
    return im


# ---------------------------------------------------------------------------
# 5. Confusion matrix NER (SKILL / EXPERIENCE / MAJOR / O)
#    Nguồn: outputs/benmark_glinner.txt (production small-v2.5, overlap)
# ---------------------------------------------------------------------------
def fig_ner_confusion():
    labels_row = ["SKILL", "EXPERIENCE", "MAJOR", "O (missed)"]
    labels_col = ["SKILL", "EXPERIENCE", "MAJOR", "O (spurious)"]
    cm = [
        [5771,    1,   21, 1379],
        [   1,  379,    0,   14],
        [   6,    0,  839,   21],
        [1185,   23,   15,    0],
    ]
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", BLUE])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = _heatmap(ax, cm, labels_row, labels_col,
                  "NER confusion matrix (GLiNER-Small-v2.5, overlap)", cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig_ner_confusion.png")


# ---------------------------------------------------------------------------
# 6. Confusion matrix Level classifier (6x6)
#    Nguồn: outputs/benmark_bert.txt
# ---------------------------------------------------------------------------
def fig_level_confusion():
    levels = ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "LEAD_PLUS"]
    cm = [
        [20,   8,   1,   0,   0,  0],
        [40, 111,   9,   2,   1,  0],
        [ 2,  21, 260,  13,   0,  4],
        [ 0,   2,  23, 118,  14,  7],
        [ 0,   0,   0,   9,  50,  8],
        [ 0,   1,   6,   7,   9, 44],
    ]
    cmap = LinearSegmentedColormap.from_list("greens", ["#ffffff", GREEN])
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = _heatmap(ax, cm, levels, levels,
                  "Level classifier confusion matrix (DistilBERT)", cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig_level_confusion.png")


# ---------------------------------------------------------------------------
# 7. Baseline vs Post-FT cho model được chọn (minh họa hiệu quả fine-tune)
#    Nguồn: benmark_glinner.txt (Small-v2.5)
# ---------------------------------------------------------------------------
def fig_baseline_vs_postft():
    metrics = ["Overall F1", "nDCG@10", "SKILL F1", "EXP F1"]
    baseline = [0.1925, 0.3435, 0.2093, 0.0076]
    postft =   [0.7754, 0.9106, 0.7686, 0.8904]

    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w/2, baseline, w, label="Zero-shot baseline", color="#94a3b8")
    b2 = ax.bar(x + w/2, postft, w, label="After fine-tuning", color=PURPLE)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Effect of fine-tuning — GLiNER-Small-v2.5")
    ax.legend(loc="upper left")
    _save(fig, "fig_baseline_vs_postft.png")


if __name__ == "__main__":
    print(f"[make_thesis_figures] Xuat hinh vao: {OUT_DIR}")
    fig_gliner_benchmark()
    fig_baseline_vs_postft()
    fig_generalization_gap()
    fig_classifier_benchmark()
    fig_classifier_per_class()
    fig_ner_confusion()
    fig_level_confusion()
    print("[make_thesis_figures] Xong. Tat ca hinh da luu duoi dang PNG 300 DPI.")
