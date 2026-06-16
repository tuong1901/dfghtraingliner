# 🎯 Train — Job Description NER + Level Classification

Thư mục chứa toàn bộ pipeline **training, benchmark và evaluation** cho 2 model:

| Model | Task | Input | Output |
|-------|------|-------|--------|
| **GLiNER** | NER | Job Description (text) | Spans: `{SKILL, EXPERIENCE}` |
| **Level Classifier** | Text Classification | Job Description (text) | Level: `INTERN → DIRECTOR` |

---

## 📁 Cấu trúc thư mục

```
train/
├── config.yaml                       ← ⭐ File cấu hình tổng (chỉnh ở đây, không sửa code)
├── config_medium_v25.yaml            ← File cấu hình chuyên biệt cho GLiNER Medium v2.5
├── config_medium_v25_major.yaml      ← File cấu hình train GLiNER Medium v2.5 kèm nhãn MAJOR
├── config_small_v25_major.yaml       ← File cấu hình train GLiNER Small v2.5 kèm nhãn MAJOR
├── config_debug.yaml                 ← File cấu hình test nhanh (dry-run)
│
├── train_master.py                   ← Script điều phối chính (chạy file này)
├── train_gliner.py                   ← Pipeline train GLiNER (single model, full epochs)
├── train_classifier.py               ← Pipeline train Level Classifier (single model)
│
├── benchmark_gliner.py               ← Benchmark GLiNER: Baseline→FT→Post-eval (F1+nDCG@5,10)
├── benchmark_classifier.py           ← Benchmark Classifier: so sánh BERT/RoBERTa/DeBERTa/...
│
├── utils.py                          ← Hàm tiện ích dùng chung (load, seed, device, ...)
├── requirements.txt                  ← pip install -r requirements.txt
│
├── README.md                         ← File này
├── documentation.md                  ← Tài liệu chi tiết từng hàm
│
└── data_test/                        ← Thư mục chứa dữ liệu test 1000 dòng và các kịch bản đánh giá
    ├── data_xin_1000_dong.xlsx       ← File Excel input 1000 dòng JD test
    ├── data_xin_1000_dong_predicted.xlsx ← File Excel kết quả dự đoán của GLiNER
    ├── data_xin_1000_dong_gold.json  ← File JSON chứa nhãn vàng chuẩn (DeepSeek V3)
    ├── test_model.py                 ← Chạy mô hình dự đoán hàng loạt thực thể ra Excel
    ├── benchmark_predicted.py        ← Đối chiếu dự đoán với Gold và đánh giá (Exact/Overlap)
    └── clean_gold_labels.py          ← Làm sạch và loại bỏ chồng lấn nhãn trên tập Gold
```


---

## 🚀 Cách chạy nhanh

```bash
cd d:\OCR\New_folder\train
pip install -r requirements.txt

# Train đầy đủ (GLiNER + Classifier theo config.yaml)
python train_master.py

# Chỉ train 1 model
python train_master.py --gliner-only
python train_master.py --classifier-only

# Benchmark (so sánh nhiều architecture)
python train_master.py --benchmark-gliner      # GLiNER variants
python train_master.py --benchmark             # BERT-family Classifier

# Kiểm tra thư viện
python train_master.py --check-deps
```

---

## 🔬 Benchmark GLiNER — Chi tiết đầy đủ

### Mục tiêu
So sánh các kiến trúc GLiNER khác nhau để tìm model **tốt nhất cho bài toán NER trên Job Description**.

### Pipeline mỗi model (3 bước)

```
┌─────────────────────────────────────────────────────────────────┐
│  Với mỗi GLiNER architecture:                                   │
│                                                                 │
│  STEP 1: Đánh giá BASELINE                                      │
│          ↓ Load pretrained model (chưa fine-tune)               │
│          ↓ Inference trên val set                               │
│          ↓ Tính F1 + nDCG@5 + nDCG@10                          │
│          ↓ Lưu baseline_metrics.json                            │
│                                                                 │
│  STEP 2: Fine-tune GLiNER                                       │
│          ↓ Train trên cleaned_dataset.json                      │
│          ↓ Lưu best checkpoint (theo eval_loss)                 │
│                                                                 │
│  STEP 3: Đánh giá POST-FINETUNE                                 │
│          ↓ Load best checkpoint                                 │
│          ↓ Inference trên val set                               │
│          ↓ Tính F1 + nDCG@5 + nDCG@10                          │
│          ↓ In bảng so sánh Baseline vs Post-FT                  │
│          ↓ Lưu eval_metrics.json                                │
└─────────────────────────────────────────────────────────────────┘

Sau khi tất cả model xong:
  → Tìm model tốt nhất (best Post-FT F1)
  → Chạy Full Pipeline Analysis trên model tốt nhất:
      - Sensitivity analysis: threshold 0.3 / 0.4 / 0.5 / 0.6
      - Báo cáo đầy đủ best_model_full_pipeline.txt
```

### Các model được benchmark

| # | Tên | HuggingFace ID | Encoder | Params | Ghi chú |
|---|-----|----------------|---------|--------|---------|
| 1 | GLiNER-Small-v2.1 | `urchade/gliner_small-v2.1` | DeBERTa-xsmall | 22M | Nhanh nhất |
| 2 | GLiNER-Medium-v2.1 | `urchade/gliner_medium-v2.1` | DeBERTa-base | 86M | ⭐ Cân bằng |
| 3 | GLiNER-Large-v2.1 | `urchade/gliner_large-v2.1` | DeBERTa-large | 304M | Chính xác nhất |
| 4 | GLiNER-Small-v2.5 | `gliner-community/gliner_small-v2.5` | DeBERTa-v3-xsmall | 22M | Data mới hơn |
| 5 | GLiNER-Medium-v2.5 | `gliner-community/gliner_medium-v2.5` | DeBERTa-v3-base | 86M | ⭐ Mới + cân bằng |
| 6 | GLiNER-Multi | `urchade/gliner_multi-v2.1` | XLM-RoBERTa | 270M | Multilingual Vi+En |

> **Ghi chú:** GLiNER-Multi tắt mặc định (`enabled: false`). Bật trong `config.yaml` nếu dataset có nhiều tiếng Việt.

### Metrics đánh giá

#### F1 / Precision / Recall (Exact Span Match)
- **Exact match**: span `(start, end, label)` phải khớp hoàn toàn ở char-level
- Tính cho từng entity type riêng: `SKILL_F1`, `EXPERIENCE_F1`
- Tính tổng hợp: `Overall_F1`

#### nDCG@5 và nDCG@10 (Normalized Discounted Cumulative Gain)

Đo **chất lượng ranking** của predictions theo confidence score — model tốt sẽ đặt những span đúng ở đầu danh sách.

```
Với mỗi document:
  1. Sort predictions theo confidence score (giảm dần)
  2. Label mỗi pred: rel=1 nếu exact match với gold, rel=0 nếu sai
  3. DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
  4. IDCG@K = DCG lý tưởng (tất cả gold đúng ở vị trí đầu)
  5. nDCG@K = DCG@K / IDCG@K  ∈ [0.0, 1.0]

Macro-average qua tất cả document → nDCG@5, nDCG@10
```

| Metric | Ý nghĩa | Khi nào dùng |
|--------|---------|--------------|
| `nDCG@5` | Ranking trong top-5 (strict) | Khi chỉ cần top predictions rất chính xác |
| `nDCG@10` | Ranking trong top-10 (phổ biến) | Đánh giá tổng thể chất lượng ranking |

> `nDCG = 1.0` → model luôn đặt span đúng trước span sai  
> `nDCG = 0.0` → model không có khả năng phân biệt span đúng/sai

#### Inference Speed
- Đo `samples/sec` trên val set
- Quan trọng cho production deployment

### Bảng kết quả đầu ra (ví dụ)

```
Model                     Stage    F1    nDCG@5 nDCG@10  SKILL_F1  EXP_F1    Time
──────────────────────────────────────────────────────────────────────────────────
GLiNER-Medium-v2.5   BASE     0.6120  0.7800  0.7650   0.6540   0.4800
                     POST-FT  0.8340  0.9120  0.9050   0.8680   0.7450   12m 30s
                     Δ        +0.2220 +0.1320 +0.1400  +0.2140  +0.2650
──────────────────────────────────────────────────────────────────────────────────
GLiNER-Large-v2.1    BASE     0.6450  0.8100  0.7980   0.6890   0.5100
                     POST-FT  0.8120  0.9050  0.8970   0.8430   0.7290   28m 15s
                     Δ        +0.1670 +0.0950 +0.0990  +0.1540  +0.2190
──────────────────────────────────────────────────────────────────────────────────
```

### Output files

```
outputs/benchmark_gliner/
├── GLiNER_Small_v2_1/
│   ├── baseline_metrics.json       ← F1 + nDCG@5 + nDCG@10 TRƯỚC fine-tune
│   ├── best_model/
│   │   ├── eval_metrics.json       ← F1 + nDCG@5 + nDCG@10 SAU fine-tune
│   │   ├── entity_types.json
│   │   └── [model weights]
│   └── [training checkpoints]
├── GLiNER_Medium_v2_1/ ...
├── GLiNER_Large_v2_1/ ...
├── GLiNER_Small_v2_5/ ...
├── GLiNER_Medium_v2_5/ ...
│
├── benchmark_gliner_results.csv    ← Tổng hợp tất cả model (Excel-friendly)
├── benchmark_gliner_results.txt    ← Bảng ASCII dễ đọc
├── best_model_full_pipeline.txt    ← Phân tích đầy đủ model tốt nhất
└── best_model_full_pipeline.json   ← JSON cho programmatic access
```

---

## 📊 Benchmark Classifier — Chi tiết

### Mục tiêu
So sánh các BERT-family model để tìm architecture tốt nhất cho **Level Classification**.

### Pipeline
Mỗi model được train trên cùng data split, evaluate bằng Accuracy + Weighted F1.

### Các model benchmark

| # | Tên | HuggingFace ID | Params | Đặc điểm |
|---|-----|----------------|--------|-----------|
| 1 | BERT-base | `bert-base-uncased` | 110M | Baseline chuẩn |
| 2 | DistilBERT | `distilbert-base-uncased` | 67M | Nhanh 2x, nhẹ hơn |
| 3 | RoBERTa-base | `roberta-base` | 125M | Thường tốt hơn BERT |
| 4 | DeBERTa-v3-small | `microsoft/deberta-v3-small` | 44M | SOTA nhỏ |
| 5 | ELECTRA-small | `google/electra-small-discriminator` | 14M | Rất nhỏ, nhanh |
| 6 | PhoBERT-base-v2 | `vinai/phobert-base-v2` | 135M | Vietnamese (tắt mặc định) |

### Metrics

| Metric | Mô tả |
|--------|-------|
| Accuracy | Tỉ lệ dự đoán đúng |
| Weighted F1 | F1 có trọng số theo số sample mỗi class |
| Val Loss | Cross-entropy loss trên val set |
| Training Time | Tổng thời gian train |

### Truncation Strategy cho JD dài (> 512 token)

```
head+tail (mặc định, tốt nhất):
  [128 token đầu] + [384 token cuối]
       ↑                    ↑
  Title, Tags          Requirements
  Job description      Experience
```

### Output files

```
outputs/
├── classifier/best_model/          ← Model từ train_classifier.py (single)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── label_map.json
└── benchmark/                      ← Kết quả từ benchmark_classifier.py
    ├── BERT_base/best_model/
    ├── DistilBERT/best_model/
    ├── RoBERTa_base/best_model/
    ├── DeBERTa_v3_small/best_model/
    ├── ELECTRA_small/best_model/
    ├── benchmark_results.csv
    └── benchmark_results.txt
```

---

## ⚙️ Cấu hình `config.yaml`

### Bật/tắt task

```yaml
run:
  train_gliner: true           # Train GLiNER (single model)
  train_classifier: true       # Train Classifier (single model)
  run_benchmark: false         # Benchmark Classifier (6 models)
  run_benchmark_gliner: false  # Benchmark GLiNER (5 models + baseline+FT)
  seed: 42
```

### Config benchmark GLiNER quan trọng

```yaml
benchmark_gliner:
  num_epochs: 3           # Epoch benchmark (ít hơn để nhanh)
  max_length: 1024        # JD dài → cần cao
  eval_threshold: 0.5     # Threshold cho F1
  low_threshold: 0.1      # Threshold cho nDCG ranking
  ndcg_ks: [5, 10]        # Tính nDCG@5 và nDCG@10
  run_best_model_pipeline: true  # Full analysis model tốt nhất
  analyze_data: true      # Phân tích phân bố entity
```

### Bật/tắt từng model

Trong `benchmark_gliner.models`, đặt `enabled: true/false` cho từng model:

```yaml
  models:
    - name: "GLiNER-Small-v2.1"
      enabled: true           # ← bật
    - name: "GLiNER-Multi"
      enabled: false          # ← tắt (multilingual, chậm hơn)
```

---

## 🛠️ CLI Flags đầy đủ

```
python train_master.py [options]

Options:
  --config PATH           File config (mặc định: config.yaml)
  --gliner-only           Chỉ train GLiNER
  --classifier-only       Chỉ train Level Classifier
  --benchmark             Benchmark Classifier (6 BERT-family)
  --benchmark-gliner      Benchmark GLiNER (5 architectures, Baseline+FT)
  --check-deps            Kiểm tra thư viện, không train
  --no-test               Bỏ qua quick test sau train
```

---

## 📦 Dataset format

File `cleaned_dataset.json` (tạo bởi các script ở thư mục cha):

```json
{
    "text": "We are looking for a Senior Python Developer...",
    "label": [
        [8, 14, "SKILL"],
        [35, 41, "SKILL"],
        [103, 120, "EXPERIENCE"]
    ],
    "level": "SENIOR"
}
```

- `text`: Job Description nguyên bản
- `label`: List spans `[start_char, end_char, entity_type]`
  - `SKILL`: Kỹ năng kỹ thuật (Python, Docker, React...)
  - `EXPERIENCE`: Số năm kinh nghiệm ("3+ years", "Minimum 5 years"...)
  - ❌ `MAJOR` đã bị loại khỏi GLiNER training
- `level`: Cấp bậc job — 1 trong 10 class:
  `INTERN | FRESHER | JUNIOR | MIDDLE | SENIOR | LEAD | MANAGER | DIRECTOR | EXPERT | UNKNOWN`

---

## 📋 Requirements

```
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
gliner>=0.1.12
scikit-learn>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
```

```bash
pip install -r requirements.txt
```

---

## 🔬 Hàm quan trọng (Quick Reference)

### `benchmark_gliner.py`

| Hàm | Tác dụng |
|-----|----------|
| `compute_ndcg_at_k(preds, golds, k)` | Tính nDCG@K cho 1 document |
| `compute_ndcg_corpus(preds_list, golds_list, k)` | Macro-avg nDCG@K qua corpus |
| `compute_ner_metrics(preds, golds, entity_types)` | F1/P/R per entity type + overall |
| `evaluate_gliner(model, val_samples, ..., ndcg_ks=[5,10])` | Inference + F1 + nDCG@5 + nDCG@10 |
| `run_single_gliner_benchmark(...)` | Baseline → Fine-tune → Post-eval cho 1 model |
| `run_best_model_full_pipeline(...)` | Threshold sensitivity analysis cho model tốt nhất |
| `run_all_gliner_benchmarks(cfg)` | Điều phối toàn bộ benchmark |
| `analyze_entity_distribution(...)` | Phân tích phân bố entity, khuyến nghị config |

### `benchmark_classifier.py`

| Hàm | Tác dụng |
|-----|----------|
| `run_single_benchmark(model_cfg, ...)` | Train + eval 1 BERT-family model |
| `run_all_benchmarks(cfg)` | Duyệt tất cả model enabled |
| `print_results_table(results)` | Bảng ASCII so sánh Accuracy/F1/Time |
| `save_results(results, output_dir)` | Lưu CSV + TXT |

### `utils.py`

| Hàm | Tác dụng |
|-----|----------|
| `load_config(path)` | Load YAML → dict |
| `set_seed(seed)` | Fix random seed Python/NumPy/PyTorch |
| `load_dataset(path, val_ratio, ...)` | Load JSON → shuffle → split train/val |
| `check_device()` | Detect GPU/CPU, in tên GPU |
| `format_time(seconds)` | `3661` → `"1h 1m 1s"` |
| `print_banner(title)` | In tiêu đề đẹp |

---

## ❓ FAQ

**Q: Tại sao benchmark GLiNER tốn RAM nhiều?**  
A: DeBERTa-large (304M) cần ~6-8GB VRAM. Giảm `train_batch_size: 4` và tăng `gradient_accumulation_steps: 4`.

**Q: nDCG@5 khác gì nDCG@10?**  
A: nDCG@5 chỉ xét top-5 predictions (strict hơn), nDCG@10 xét top-10 (dễ hơn). Thường `nDCG@5 ≤ nDCG@10`.

**Q: Tại sao `low_threshold=0.1` khi tính nDCG?**  
A: Để có đủ candidates cho ranking. Nếu dùng threshold=0.5, sẽ lọc bỏ nhiều predictions → không đủ để xếp hạng top-10.

**Q: Khi nào nên bật GLiNER-Multi?**  
A: Khi dataset có nhiều JD tiếng Việt (> 30%). Bật bằng `enabled: true` trong config.

**Q: Baseline F1 thấp có bình thường không?**  
A: Có! Pretrained GLiNER chưa biết domain Job Description. Delta (Δ) sau fine-tune mới là quan trọng.
