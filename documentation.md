# Tài liệu thư mục `c:\Users\loiha\Videos\dfghtraingliner`

## Tổng quan
Thư mục này chứa toàn bộ code training và benchmark cho **2 model**:
1. **GLiNER NER Model** – Trích xuất thông tin từ Job Description
   - **Đầu vào**: `text` (chuỗi Job Description thuần)
   - **Đầu ra**: Spans `[start, end, label]` với label ∈ `{SKILL, EXPERIENCE}`
2. **Level Classifier** – Phân loại cấp bậc công việc
   - **Đầu vào**: `text` (chuỗi Job Description thuần)
   - **Đầu ra**: `level` ∈ `{FRESHER, JUNIOR, MIDDLE, SENIOR, UNKNOWN}` (các lớp thiểu số khác như INTERN, LEAD, MANAGER, DIRECTOR, EXPERT tự động được map về UNKNOWN)

Dataset đầu vào: `../cleaned_dataset.json`

---

## Cách sử dụng nhanh

```bash
cd d:\OCR\New_folder\train

# 1. Cài thư viện
pip install -r requirements.txt

# 2. Chỉnh config.yaml (chọn model nào train, set hyperparameters)

# 3. Chạy
python train_master.py                      # Train theo config.yaml (GLiNER + Classifier)
python train_master.py --gliner-only        # Chỉ train GLiNER
python train_master.py --classifier-only    # Chỉ train Level Classifier
python train_master.py --benchmark-gliner   # So sánh nhiều GLiNER architecture
python train_master.py --benchmark          # So sánh nhiều Classifier (BERT/RoBERTa/...)
python train_master.py --check-deps         # Kiểm tra thư viện trước

# Hoặc chạy riêng từng script
python train_gliner.py
python train_classifier.py
python benchmark_gliner.py
python benchmark_classifier.py
```

---

## Các tệp

### `config.yaml`
File cấu hình tổng. **Chỉ cần chỉnh file này**, không cần sửa code.

**Các section chính:**
| Section | Tác dụng |
|---------|----------|
| `run` | Bật/tắt task: `train_gliner`, `train_classifier`, `run_benchmark`, `run_benchmark_gliner` |
| `data` | Đường dẫn dataset, val/train ratio, giới hạn số sample |
| `gliner` | Hyperparameter cho GLiNER single-model train |
| `classifier` | Hyperparameter cho Level Classifier single-model train |
| `benchmark` | Benchmark Classifier: 6 model BERT-family |
| `benchmark_gliner` | Benchmark GLiNER: 6 GLiNER variant (Small/Medium/Large/v2.5/Multi) |

---

### `train_master.py`
**Script điều phối chính** – File duy nhất cần chạy.

**Hàm/logic chính:**
- **`main()`**: Parse CLI args → load config → kiểm tra deps → gọi từng pipeline → in tóm tắt
- **`check_dependencies(run_gliner, run_classifier)`**: Kiểm tra thư viện (torch, transformers, gliner...). In hướng dẫn `pip install` nếu thiếu
- **`print_config_summary(cfg)`**: In bảng tóm tắt config trước khi train
- **`resolve_dataset_path(cfg, config_path)`**: Resolve đường dẫn dataset tương đối → tuyệt đối
- **`resolve_output_dirs(cfg, config_path)`**: Resolve output_dir tương đối → tuyệt đối

**CLI flags:**
| Flag | Tác dụng |
|------|----------|
| `--config path` | Dùng file config khác (mặc định: config.yaml) |
| `--gliner-only` | Chỉ train GLiNER |
| `--classifier-only` | Chỉ train Classifier đơn |
| `--benchmark` | Benchmark nhiều Classifier model (BERT/RoBERTa/DeBERTa/...) |
| `--benchmark-gliner` | **Benchmark nhiều GLiNER architecture (Small/Medium/Large/v2.5/Multi)** |
| `--check-deps` | Chỉ kiểm tra thư viện |
| `--no-test` | Bỏ qua quick test sau train |

---

### `train_gliner.py`
**Pipeline train GLiNER NER model** (single model, đầy đủ epochs)

**Input/Output:**
- Đầu vào: `text` (Job Description)
- Đầu ra: Spans `[start, end, label]` với label ∈ `{SKILL, EXPERIENCE}` (**bỏ MAJOR**)

**Hàm chính:**
- **`prepare_gliner_samples(data, entity_types, max_length)`**: Chuyển `cleaned_dataset.json` → GLiNER format `{text, entities: [{start, end, label}]}`. Lọc tự động label không trong entity_types, kiểm tra bounds, bỏ span rỗng
- **`get_gliner_data_collator(model)`**: Tự động nhận diện phiên bản `gliner` (v0.1.x hay v0.2.x) để trả về class data collator thích hợp (`DataCollator`, `SpanDataCollator`, hoặc `BiEncoderSpanDataCollator`)
- **`train_gliner(cfg)`**: Load GLiNER từ HuggingFace, setup Trainer riêng của GLiNER (dùng dynamic collator), train với fp16 nếu có GPU, lưu final model
- **`quick_test_gliner(model_dir, entity_types)`**: Test nhanh 1 câu mẫu sau train

**Output:** `./outputs/gliner/final_model/` + `entity_types.json`

**Lưu ý:**
- GLiNER có Trainer riêng, không dùng HuggingFace Trainer
- Dataset là list dict Python thuần
- Zero-shot: có thể predict entity type mới mà không retrain

---

### `train_classifier.py`
**Pipeline train BERT/RoBERTa/DeBERTa Level Classifier**

**Input/Output:**
- Đầu vào: `text` (chuỗi Job Description thuần)
- Đầu ra: `level` label (multi-class classification, 10 lớp)

**Hàm/class chính:**
- **`JobLevelDataset`** (class): PyTorch Dataset. `__getitem__` → `{input_ids, attention_mask, labels}`
  - **`_tokenize(text)`**: 3 chiến lược truncation:
    - `"head"`: lấy max_length token đầu
    - `"tail"`: lấy max_length token cuối (bắt requirements)
    - `"head+tail"`: 128 đầu + phần cuối (tốt nhất, mặc định)
- **`train_classifier(cfg)`**: Training loop với trọng số lớp (**Weighted CrossEntropyLoss** dùng `class_weights` từ training set) để giải quyết mất cân bằng dữ liệu. Tích hợp AdamW + Linear scheduler + warmup + best-model checkpoint.
- **`quick_test_classifier(model_dir, level_labels)`**: Test 4 câu mẫu với levels khác nhau

**Output:** `./outputs/classifier/best_model/` + `label_map.json`

---

### `benchmark_classifier.py`
**Script so sánh nhiều BERT-family model** để chọn Classifier tốt nhất.

**Các model benchmark (mặc định):**
| Model | HuggingFace ID | Đặc điểm |
|-------|----------------|----------|
| BERT-base | `bert-base-uncased` | Baseline chuẩn |
| DistilBERT | `distilbert-base-uncased` | Nhanh 2x, nhẹ hơn |
| RoBERTa-base | `roberta-base` | Thường tốt hơn BERT |
| DeBERTa-v3-small | `microsoft/deberta-v3-small` | SOTA nhỏ |
| ELECTRA-small | `google/electra-small-discriminator` | Rất nhỏ, nhanh |
| PhoBERT-base-v2 | `vinai/phobert-base-v2` | Vietnamese (tắt mặc định) |

**Hàm chính:**
- **`run_single_benchmark(model_cfg, ...)`**: Train + evaluate 1 model → `{accuracy, f1_weighted, val_loss, train_time, n_params, status}`
- **`run_all_benchmarks(cfg)`**: Loop qua tất cả model enabled, tổng hợp kết quả
- **`print_results_table(results)`**: In bảng ASCII so sánh + highlight winner
- **`save_results(results, output_dir)`**: Lưu `benchmark_results.csv` + `benchmark_results.txt`

---

### `benchmark_gliner.py`
**Script so sánh nhiều GLiNER architecture** để chọn model NER tốt nhất cho Job Description.

Thiết kế phù hợp với data:
- SKILL (span ngắn 1-5 word) + EXPERIENCE (span trung bình 5-15 word)
- JD text dài → `max_length=1024`, GLiNER dùng sliding window
- Phân tích phân bố entity tự động trước khi benchmark

**Các GLiNER variant benchmark:**
| Model | HuggingFace ID | Encoder | Đặc điểm |
|-------|----------------|---------|-----------|
| GLiNER-Small-v2.1 | `urchade/gliner_small-v2.1` | DeBERTa-xsmall (22M) | Nhanh nhất |
| GLiNER-Medium-v2.1 | `urchade/gliner_medium-v2.1` | DeBERTa-base (86M) | ⭐ Recommend |
| GLiNER-Large-v2.1 | `urchade/gliner_large-v2.1` | DeBERTa-large (304M) | Chính xác nhất |
| GLiNER-Small-v2.5 | `gliner-community/gliner_small-v2.5` | DeBERTa-v3-xsmall | Mới nhất, nhỏ |
| GLiNER-Medium-v2.5 | `gliner-community/gliner_medium-v2.5` | DeBERTa-v3-base | Mới nhất, recommend |
| GLiNER-Multi | `urchade/gliner_multi-v0.1` | XLM-RoBERTa (270M) | Multilingual Vi+En |

**Hàm chính:**
- **`compute_ndcg_at_k(preds, golds, k)`**: Tính nDCG@K cho 1 document (char-level exact span match). P/Rố predictions theo score giảm dần, relevance=1 nếu exact match với gold, tính DCG/IDCG
- **`compute_ndcg_corpus(preds_list, golds_list, k)`**: Macro-average nDCG@K qua toàn bộ corpus
- **`compute_ner_metrics(predictions, gold_labels, entity_types)`**: Tính P/R/F1 per entity type dùng exact span match (char-level). Không dùng GLiNER’s built-in eval để có per-type detail
- **`evaluate_gliner(model, val_samples, entity_types, threshold, batch_size, ndcg_ks=[5,10], low_threshold)`**: Batch inference + tính metrics. Dùng 2 threshold: `threshold` (strict, cho F1) và `low_threshold` (loose, cho nDCG ranking). Trả về `{overall_f1, SKILL_f1, EXPERIENCE_f1, ndcg_at_5, ndcg_at_10, infer_speed}`
- **`analyze_entity_distribution(samples, entity_types)`**: Phân tích phân bố entity trong dataset (count per type, span length, text length) → khuyến nghị config
- **`run_single_gliner_benchmark(model_cfg, ...)`**: Pipeline đầy đủ cho 1 model: (1) Baseline eval (F1+nDCG@5+nDCG@10 trước FT) → (2) Fine-tune → (3) Post-FT eval → (4) In bảng so sánh ΔF1 + ΔnDCG
- **`run_best_model_full_pipeline(best_result, ...)`**: Threshold sensitivity analysis (0.3/0.4/0.5/0.6) trên model tốt nhất, lưu báo cáo đầy đủ
- **`run_all_gliner_benchmarks(cfg)`**: Dưyệt tất cả model enabled, tổng hợp bảng kết quả, in + lưu
- **`print_gliner_results_table(results, entity_types)`**: Bảng ASCII per-type F1 + nDCG@5 + nDCG@10 + Delta so sánh Baseline vs Post-FT
- **`save_gliner_results(results, entity_types, output_dir)`**: Lưu `benchmark_gliner_results.csv` (cả Baseline + Post-FT fields) + `.txt`

**Output:**
```
outputs/benchmark_gliner/
  GLiNER_Small_v2_1/best_model/
    eval_metrics.json      ← P/R/F1 per entity type + infer speed
    entity_types.json
  GLiNER_Medium_v2_1/best_model/
  GLiNER_Large_v2_1/best_model/
  GLiNER_Small_v2_5/best_model/
  GLiNER_Medium_v2_5/best_model/
  benchmark_gliner_results.csv    ← import vào Excel
  benchmark_gliner_results.txt    ← bảng ASCII dễ đọc
```

---

### `utils.py`
**Hàm tiện ích dùng chung** giữa tất cả pipeline.

| Hàm | Tác dụng |
|-----|----------|
| `load_config(config_path)` | Đọc YAML → dict |
| `set_seed(seed)` | Fix random seed (Python, NumPy, PyTorch) |
| `load_dataset(path, val_ratio, max_samples, seed, level_labels)` | Load JSON, lọc và gộp các level thiểu số không nằm trong config về `'UNKNOWN'`, split train/val |
| `normalize_level(level_str, level_labels)` | Chuỗi level → index |
| `format_time(seconds)` | Số giây → "Xh Ym Zs" |
| `print_banner(title)` | In tiêu đề đẹp |
| `check_device()` | Kiểm tra GPU/CPU |

---

### `requirements.txt`
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

---

## Cấu trúc output sau khi train/benchmark

```
train/
  outputs/
    gliner/
      checkpoint-500/
      final_model/
        config.json
        pytorch_model.bin
        entity_types.json      # ["SKILL", "EXPERIENCE"]
    classifier/
      best_model/
        config.json
        pytorch_model.bin
        label_map.json         # {"level_labels": [...], "id2label": {...}}
    benchmark/
      BERT_base/best_model/
      DistilBERT/best_model/
      RoBERTa_base/best_model/
      DeBERTa_v3_small/best_model/
      ELECTRA_small/best_model/
      benchmark_results.csv   ← so sánh tổng hợp
      benchmark_results.txt
```

---

## Ghi chú kỹ thuật

### Dataset format
```json
{
    "text": "Job description text...",
    "label": [[8, 16, "SKILL"], [1380, 1401, "EXPERIENCE"]],
    "level": "SENIOR"
}
```

### GLiNER: Entity Types được train
- ✅ `SKILL`: Kỹ năng kỹ thuật (Python, React, Docker...)
- ✅ `EXPERIENCE`: Số năm kinh nghiệm ("3+ years", "Minimum 5 years"...)
- ❌ `MAJOR`: **Không train** – đã bỏ khỏi GLiNER entity_types

### Classifier: Level Classes (5 lớp chính)
`FRESHER` → `JUNIOR` → `MIDDLE` → `SENIOR` → `UNKNOWN` (Các nhãn khác như `INTERN`, `LEAD`, `MANAGER`, `DIRECTOR`, `EXPERT` tự động được chuẩn hóa về `UNKNOWN` lúc load dataset).

### Truncation strategy
JD thường > 512 token. Chiến lược `head+tail` hiệu quả nhất:
- **128 token đầu**: tags, job title, mô tả chính
- **384 token cuối**: yêu cầu kinh nghiệm, học vấn (key để dự đoán level)

---

### Lịch sử cập nhật tài liệu và mã nguồn
- **2026-06-12**: Sửa lỗi `TypeError: evaluate_gliner() got an unexpected keyword argument 'ndcg_k'` trong [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py) bằng cách đổi tham số truyền vào từ `ndcg_k=ndcg_k` thành `ndcg_ks=[ndcg_k]` để khớp với chữ ký của hàm `evaluate_gliner`.
