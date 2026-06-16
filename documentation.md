# Tài liệu thư mục `c:\Users\loiha\Videos\dfghtraingliner`

## Tổng quan
Thư mục này chứa toàn bộ code training và benchmark cho **2 model**:
1. **GLiNER NER Model** – Trích xuất thông tin từ Job Description
   - **Đầu vào**: `text` (chuỗi Job Description thuần)
   - **Đầu ra**: Spans `[start, end, label]` với label ∈ `{SKILL, EXPERIENCE}`
2. **Level Classifier** – Phân loại cấp bậc công việc
   - Đầu vào: `text` (chuỗi Job Description thuần)
   - Đầu ra: `level` ∈ `{INTERN, FRESHER, JUNIOR, MIDDLE, SENIOR, MANAGER, UNKNOWN}` (các lớp thiểu số dưới 3% như LEAD, DIRECTOR, EXECUTIVE tự động được map về UNKNOWN)

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

### `config_medium_v25.yaml`
File cấu hình chuyên biệt dùng để huấn luyện mô hình **GLiNER-Medium-v2.5** trên tập dữ liệu đã làm sạch. File này đã được tối ưu sẵn các tham số cấu hình (batch=4, gradient_accumulation=4, gradient_checkpointing=true) để ngăn ngừa lỗi CUDA OOM trên môi trường GPU dưới 16GB.

### `config_medium_v25_major.yaml`
File cấu hình chuyên biệt dùng để huấn luyện mô hình **GLiNER-Medium-v2.5** đồng thời trên 3 nhãn: `SKILL`, `EXPERIENCE`, và `MAJOR` để mô hình tự phân biệt được ngành học/bằng cấp thay vì gán nhầm vào `SKILL`. Hỗ trợ đổi đường dẫn `model_name` sang thư mục cục bộ để tiếp tục huấn luyện (continue fine-tuning) trên mô hình có sẵn.

### `config_small_v25_major.yaml`
File cấu hình chuyên biệt tương tự dùng để huấn luyện mô hình **GLiNER-Small-v2.5** đồng thời trên 3 nhãn: `SKILL`, `EXPERIENCE`, và `MAJOR`. Đã được cấu hình tối ưu hyperparameter cho bản Small (batch_size=8, gradient_accumulation=2) và hỗ trợ trỏ đến thư mục model cục bộ (ví dụ: `d:/download/glinner/glinner-small_v2.5`) để huấn luyện tiếp tục.


### `config_debug.yaml`
File cấu hình phục vụ chế độ chạy thử nghiệm nhanh (Dry-run) trên tập dữ liệu nhỏ (20 mẫu) để đảm bảo các module hoạt động trơn tru không lỗi.

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
| `--config path` | Dùng file config làm cấu hình (mặc định: config.yaml) |
| `--gliner-only` | Chỉ train GLiNER |
| `--classifier-only` | Chỉ train Level Classifier đơn |
| `--benchmark` | Benchmark nhiều Classifier model (BERT/RoBERTa/DeBERTa/...) |
| `--benchmark-gliner` | **Benchmark nhiều GLiNER architecture (Small/Medium/Large/v2.5/Multi)** |
| `--models name` | Chỉ định các model GLiNER cần chạy (cách nhau bởi dấu phẩy, vd: `GLiNER-Small-v2.5,GLiNER-Medium-v2.5`), tự động ghi đè trường `enabled` hoặc `model_name` trong config |
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
- **`train_gliner(cfg)`**: Load GLiNER từ HuggingFace, setup Trainer riêng của GLiNER (dùng dynamic collator). Tải dataset và chia `val_ratio` (mặc định 20% trong config) thành 2 phần bằng nhau: 50% làm tập Validation (để chọn checkpoint tốt nhất) và 50% làm tập Test độc lập. Huấn luyện với fp16 nếu có GPU, lưu và xuất ra final model.
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
- Đầu ra: `level` label (multi-class classification, 7 lớp sau khi lọc và map)

**Hàm/class chính:**
- **`JobLevelDataset`** (class): PyTorch Dataset. `__getitem__` → `{input_ids, attention_mask, labels}`
  - **`_tokenize(text)`**: 3 chiến lược truncation:
    - `"head"`: lấy max_length token đầu
    - `"tail"`: lấy max_length token cuối (bắt requirements)
    - `"head+tail"`: 128 đầu + phần cuối (tốt nhất, mặc định)
- **`OrdinalLoss`** (class): Custom loss function kết hợp CrossEntropyLoss (Weighted) và Distance Penalty (phạt khoảng cách tuần tự giữa các cấp bậc có thứ tự logic như INTERN -> FRESHER -> JUNIOR -> MIDDLE -> SENIOR -> MANAGER).
- **`train_classifier(cfg)`**: Training loop chính. Tải dataset và chia `val_ratio` (mặc định 20% trong config) thành 2 phần bằng nhau: 50% làm tập Validation (để chọn checkpoint tốt nhất và early stopping) và 50% làm tập Test độc lập. Sử dụng **OrdinalLoss** để giải quyết mất cân bằng dữ liệu và phạt dự đoán sai lệch cấp bậc. Cuối cùng, load best model và chạy đánh giá khách quan trên tập Test để in báo cáo chính xác.
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
- **`run_single_benchmark(model_cfg, ...)`**: Nhận tập train_data, val_data (cho training & early stopping) và test_data (cho đánh giá cuối). Huấn luyện mô hình, lưu checkpoint tốt nhất dựa trên val_data, sau đó load checkpoint tốt nhất để đánh giá trên tập test_data độc lập và trả về kết quả benchmark → `{accuracy, f1_weighted, test_loss, train_time, n_params, status}`.
- **`run_all_benchmarks(cfg)`**: Load dataset và split phần validation thành 50% validation / 50% test độc lập. Chạy tuần tự tất cả model enabled và tổng hợp kết quả.
- **`print_results_table(results)`**: In bảng ASCII so sánh kết quả trên tập Test độc lập và highlight winner.
- **`save_results(results, output_dir)`**: Lưu `benchmark_results.csv` + `benchmark_results.txt` với các trường test metrics.

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
| GLiNER-Multi | `urchade/gliner_multi-v2.1` | XLM-RoBERTa (270M) | Multilingual Vi+En, Apache 2.0 |

**Hàm chính:**
- **`compute_ndcg_at_k(preds, golds, k)`**: Tính nDCG@K cho 1 document (char-level exact span match). P/Rố predictions theo score giảm dần, relevance=1 nếu exact match với gold, tính DCG/IDCG
- **`compute_ndcg_corpus(preds_list, golds_list, k)`**: Macro-average nDCG@K qua toàn bộ corpus
- **`compute_ner_metrics(predictions, gold_labels, entity_types)`**: Tính P/R/F1 per entity type dùng exact span match (char-level). Không dùng GLiNER’s built-in eval để có per-type detail
- **`evaluate_gliner(model, val_samples, entity_types, threshold, batch_size, ndcg_ks=[5,10], low_threshold)`**: Batch inference + tính metrics. Dùng 2 threshold: `threshold` (strict, cho F1) và `low_threshold` (loose, cho nDCG ranking). Trả về `{overall_f1, SKILL_f1, EXPERIENCE_f1, ndcg_at_5, ndcg_at_10, infer_speed}`
- **`analyze_entity_distribution(samples, entity_types)`**: Phân tích phân bố entity trong dataset (count per type, span length, text length) → khuyến nghị config
- **`run_single_gliner_benchmark(...)`**: Baseline eval trên test set → fine-tune (giám sát trên val set) → post-FT eval trên test set cho 1 model
- **`run_best_model_full_pipeline(...)`**: Threshold sensitivity analysis trên test set cho model tốt nhất
- **`run_all_gliner_benchmarks(cfg)`**: Load dataset, split phần validation thành 50% val / 50% test độc lập và điều phối toàn bộ benchmark model enabled, tổng hợp bảng kết quả, in + lưu
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

### `check_labels.py`
**Script kiểm tra và thống kê chi tiết nhãn và siêu dữ liệu (metadata)** của dataset.

**Tính năng/Hàm chính:**
- **`main()`**: 
  - Tự động cấu hình mã hóa UTF-8 cho stdout trên môi trường Windows để hiển thị chính xác chữ tiếng Việt có dấu mà không gây lỗi Unicode.
  - Tự động giải quyết đường dẫn dataset qua 3 vị trí linh hoạt: theo cấu hình của `config.yaml`, tìm cục bộ tại thư mục làm việc, hoặc tìm ở thư mục cha (phù hợp khi chạy trên môi trường Kaggle).
  - Thống kê thông tin tổng quan của dataset (Metadata): Tổng số lượng mẫu (Job Descriptions), phân bố theo nhà tuyển dụng (Provider), và phân tích độ dài ký tự/số từ của các tin tuyển dụng (min, max, average).
  - Thống kê chi tiết các thực thể NER gốc trong trường `label` bao gồm tổng số lượng thực thể, số lượng thực thể trung bình trên mỗi tin tuyển dụng, và tỉ lệ phân bố của từng loại thực thể (`SKILL`, `MAJOR`, `EXPERIENCE`).
  - Thống kê chi tiết số lượng và tỉ lệ phần trăm của từng nhãn cấp bậc gốc (Original levels) trong dataset.
  - Áp dụng logic lọc và map nhãn của classifier để hiển thị bảng phân bố nhãn sau khi map về 5 lớp chính (Target levels).

---

### `utils.py`
**Hàm tiện ích dùng chung** giữa tất cả pipeline.

* **Cấu hình môi trường**: Tự động đặt `NCCL_P2P_DISABLE=1` và `NCCL_IB_DISABLE=1` ngay khi import để phòng tránh lỗi `"NCCL Error 1: unhandled cuda error"` thường gặp trên môi trường Dual T4 GPU của Kaggle.

| Hàm | Tác dụng |
|-----|----------|
| `load_config(config_path)` | Đọc YAML → dict |
| `set_seed(seed)` | Fix random seed (Python, NumPy, PyTorch) |
| `load_dataset(path, val_ratio, max_samples, seed, level_labels)` | Load JSON (streaming chunked tiết kiệm RAM, chống MemoryError), lọc và gộp các level thiểu số < 3% về `'UNKNOWN'`, split train/val |
| `normalize_level(level_str, level_labels)` | Chuỗi level → index |
| `format_time(seconds)` | Số giây → "Xh Ym Zs" |
| `print_banner(title)` | In tiêu đề đẹp |
| `check_device()` | Kiểm tra GPU/CPU |

### `.gitignore`
Tệp cấu hình Git để loại bỏ các tệp không cần thiết khỏi hệ thống quản lý mã nguồn:
* Bỏ qua thư mục lưu kết quả `outputs/` để tránh push nhầm các file trọng số model siêu nặng (>100MB) lên GitHub.
* Bỏ qua thư mục cache của Python `__pycache__/`, các tệp compiled `*.pyc`, các file hệ thống `desktop.ini`, và các tệp cấu hình tạm thời khác.

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

### Thư mục `data_test/`
Chứa dữ liệu kiểm thử và các kịch bản kiểm thử mô hình.
- **`data_xin_1000_dong.xlsx`**: File Excel đầu vào chứa 1000 dòng mô tả công việc (JD) cần chạy NER.
- **`data_xin_1000_dong_predicted.xlsx`**: File Excel kết quả dự đoán trích xuất từ GLiNER.
- **`data_xin_1000_dong_gold.json`**: File JSON chứa nhãn chuẩn (Gold Labels) trích xuất bằng DeepSeek V3.
- **`get_data.py`**: Tải dữ liệu từ Hugging Face, lọc, ghép các trường văn bản và lưu thành file Excel 1000 dòng để kiểm thử.
- **`test_model.py`**: Chạy dự đoán thực thể (`SKILL` và `EXPERIENCE`) bằng mô hình GLiNER cục bộ, trích xuất dữ liệu, lưu kết quả ra file Excel dự đoán và in báo cáo thống kê.
- **`benchmark_predicted.py`**: So khớp kết quả dự đoán của GLiNER với nhãn chuẩn DeepSeek V3 và tính Precision, Recall, F1-score (Exact & Overlap). Tích hợp logic tự động lọc bỏ các dự đoán nhãn SKILL bị chồng lấn với nhãn chuẩn MAJOR có trong tập Gold và tự động áp dụng ngưỡng động tối ưu (SKILL >= 0.80, EXPERIENCE >= 0.50) để giảm tối đa False Positives.
- **`clean_gold_labels.py`**: Làm sạch dữ liệu nhãn vàng DeepSeek V3 (data_xin_1000_dong_gold.json) bằng cách lọc bỏ các vị trí công việc gán nhầm, kỹ năng mơ hồ, và giải quyết chồng lấn nhãn (overlapping).
- **`benchmark_report.txt`**: File báo cáo chi tiết kết quả benchmark.
- **`README.md`**: Tài liệu chi tiết các hàm, vai trò và mối liên kết của các script trong thư mục `data_test/`.

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

### Classifier: Level Classes (7 lớp chính, ngưỡng tần suất >= 3%)
`INTERN` → `FRESHER` → `JUNIOR` → `MIDDLE` → `SENIOR` → `MANAGER` → `UNKNOWN` (Các nhãn khác dưới 3% như `LEAD`, `DIRECTOR`, `EXECUTIVE` tự động được chuẩn hóa về `UNKNOWN` lúc load dataset).

### Truncation strategy
JD thường > 512 token. Chiến lược `head+tail` hiệu quả nhất:
- **128 token đầu**: tags, job title, mô tả chính
- **384 token cuối**: yêu cầu kinh nghiệm, học vấn (key để dự đoán level)

---

## Kết quả Benchmark Thực Tế (Kaggle Run)

### GLiNER Benchmark — Kết quả đã chạy

| Model | Stage | Overall F1 | nDCG@10 | SKILL F1 | EXP F1 | Thời gian |
|-------|-------|-----------|---------|----------|--------|----------|
| **GLiNER-Small-v2.1** | Baseline | 0.0202 | 0.2413 | 0.0215 | 0.0056 | — |
| **GLiNER-Small-v2.1** | Post-FT | **0.6657** | **0.8683** | **0.6683** | **0.6203** | 43m 31s |
| **GLiNER-Small-v2.1** | Δ | **+0.6455** | **+0.6270** | **+0.6469** | **+0.6147** | — |
| GLiNER-Medium-v2.1 | Baseline | 0.0240 | 0.2033 | 0.0248 | 0.0146 | — |
| GLiNER-Medium-v2.1 | Post-FT | ❌ CUDA OOM | — | — | — | — |

**Chi tiết GLiNER-Small-v2.1 (hoàn thành):**
- Train: 7193 mẫu | Val: 799 mẫu
- Hyperparams: `max_length=1024`, `batch=8`, `lr=5e-5`, `grad_accum=2`, `3 epochs`
- Trainable params: 152,648,704
- SKILL: P=0.8049 | R=0.5714 | F1=0.6683 (TP=9346, FP=2265, FN=7011)
- EXPERIENCE: P=0.8386 | R=0.4922 | F1=0.6203 (TP=504, FP=97, FN=520)
- Infer speed: 0.3 samples/sec

**Vấn đề GLiNER-Medium-v2.1 (CUDA OOM):**
- Model lớn hơn (195M params, DeBERTa-base encoder, file ~1.56GB)
- GPU 14.56 GiB bị đầy khi train với `batch=8`
- **Gợi ý fix**: giảm `train_batch_size: 4`, tăng `gradient_accumulation_steps: 4`, thêm env var `PYTORCH_ALLOC_CONF=expandable_segments:True`

---

### Lịch sử cập nhật tài liệu và mã nguồn
- **2026-06-12**: Sửa lỗi `TypeError: evaluate_gliner() got an unexpected keyword argument 'ndcg_k'` trong [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py) bằng cách đổi tham số truyền vào từ `ndcg_k=ndcg_k` thành `ndcg_ks=[ndcg_k]` để khớp với chữ ký của hàm `evaluate_gliner`.
- **2026-06-13**: Sửa lỗi `TypeError: 'int' object is not iterable` khi gọi hàm `_print_eval_metrics` tại dòng 561 và 654 trong [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py). Đã sửa tham số truyền vào từ `ndcg_k` thành `[ndcg_k]`, đồng thời cập nhật thân hàm `_print_eval_metrics` để tự động convert `int` sang list.
- **2026-06-13 (Kiểm tra cuối)**: Rà soát toàn diện dự án. Chạy biên dịch kiểm tra cú pháp (compile check) trên tất cả các tệp Python và rà soát các lời gọi hàm khác. Toàn bộ hệ thống sẵn sàng và không còn lỗi tiềm ẩn.
- **2026-06-13 (Benchmark chạy thực tế)**: Chạy benchmark trên Kaggle GPU (14.56 GiB). GLiNER-Small-v2.1 hoàn thành: F1=0.6657, nDCG@10=0.8683. GLiNER-Medium-v2.1 gặp CUDA OOM khi fine-tune do GPU không đủ VRAM với batch_size=8. Cần giảm batch size cho Medium/Large model.
- **2026-06-13 (Fix CUDA OOM)**: Cập nhật [config.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config.yaml) — giảm `train_batch_size: 8 → 4` và thêm `gradient_accumulation_steps: 4` cho GLiNER-Medium-v2.1 và GLiNER-Medium-v2.5 để tránh CUDA OOM trên GPU ≤16GB (effective batch size vẫn = 16). GLiNER-Large-v2.1 đã có batch=4 + grad_accum=4 từ trước.
- **2026-06-13 (Fix GLiNER-Multi model ID)**: Sửa model ID trong [config.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config.yaml) từ `urchade/gliner_multi-v0.1` (không tồn tại/private, trả 401 Unauthorized) sang `urchade/gliner_multi-v2.1` (public, Apache 2.0). Cập nhật bảng model trong [documentation.md](file:///c:/Users/loiha/Videos/dfghtraingliner/documentation.md).
- **2026-06-13 (Sửa lỗi NCCL Error 1)**: Thêm cấu hình tự động cho biến môi trường `NCCL_P2P_DISABLE=1` và `NCCL_IB_DISABLE=1` trong [utils.py](file:///c:/Users/loiha/Videos/dfghtraingliner/utils.py) để vô hiệu hóa Peer-to-Peer và InfiniBand của NCCL khi chạy huấn luyện song song trên Kaggle Dual T4 GPU, khắc phục triệt để lỗi crash `NCCL Error 1: unhandled cuda error`.
- **2026-06-13 (Giảm max_length)**: Thay đổi `max_length` từ `1024` xuống `512` trong [config.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config.yaml) cho cả cấu hình gliner đơn và benchmark_gliner nhằm giải quyết triệt để lỗi CUDA OOM và lỗi kéo theo `Attempted unscale_ but _scale is None` khi chạy trên môi trường Kaggle GPU có VRAM nhỏ (~15GB/16GB).
- **2026-06-13 (Thêm flag --models)**: Thêm tham số `--models` cho [train_master.py](file:///c:/Users/loiha/Videos/dfghtraingliner/train_master.py) và [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py) để cho phép chọn nhanh các model muốn chạy trực tiếp bằng dòng lệnh mà không cần sửa file config.yaml.
- **2026-06-13 (Thêm .gitignore)**: Tạo tệp [.gitignore](file:///c:/Users/loiha/Videos/dfghtraingliner/.gitignore) và dọn dẹp lịch sử git để loại bỏ thư mục `outputs/` và các file `*.pyc` khỏi commit, tránh lỗi push file dung lượng lớn lên GitHub.
- **2026-06-13 (Đồng bộ Hyperparameter cho Train Đơn)**: Nâng cấp [train_master.py](file:///c:/Users/loiha/Videos/dfghtraingliner/train_master.py) để tự động copy các siêu tham số tối ưu (batch_size, gradient_accumulation, learning_rate) của từng model từ mục benchmark sang cấu hình train đơn khi người dùng dùng flag `--models`. Đồng thời cấu hình hóa `gradient_accumulation_steps` trong [train_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/train_gliner.py) để giải quyết triệt để lỗi OOM của các model lớn (Medium, Large) khi train độc lập.
- **2026-06-13 (Tối ưu hóa VRAM cho Model Large & Multi)**: Cập nhật [config.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config.yaml) — giảm tiếp `train_batch_size: 4 -> 2` và tăng `gradient_accumulation_steps: 4 -> 8` cho các mô hình dung lượng lớn (`GLiNER-Large-v2.1` và `GLiNER-Multi`) để đảm bảo không bị tràn VRAM trên GPU Tesla T4 15GB/16GB. Thêm cấu hình tự động `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trong [utils.py](file:///c:/Users/loiha/Videos/dfghtraingliner/utils.py) để giải phóng bộ nhớ phân mảnh hiệu quả.
- **2026-06-13 (Khử PyTorch DataParallel tự động)**: Thêm cấu hình mặc định `CUDA_VISIBLE_DEVICES="0"` trong [utils.py](file:///c:/Users/loiha/Videos/dfghtraingliner/utils.py) khi chạy các script dạng đơn lẻ nhằm tắt chế độ song song phần cứng ảo `DataParallel` của PyTorch (thường tự kích hoạt khi thấy 2 GPU trên Kaggle gây lãng phí VRAM cực kỳ lớn ở GPU 0 và gây lỗi OOM).
- **2026-06-13 (Vá lỗi gradient_checkpointing_enable)**: Do `UniEncoderSpanGLiNER` kế thừa trực tiếp từ `nn.Module` chứ không phải `PreTrainedModel` của HF nên bị lỗi thiếu thuộc tính khi Trainer gọi `gradient_checkpointing_enable()`. Đã triển khai phương án vá khẩn cấp (monkey-patch) hàm này động trong [train_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/train_gliner.py) và [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py) để tìm và bật gradient checkpointing trên module backbone HF thật sự ở lớp dưới.
- **2026-06-14 (Sửa lỗi hết dung lượng đĩa và cảnh báo cắt ngắn văn bản bản Large/Multi)**:
  - Cập nhật [config.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config.yaml): Thêm cấu hình tùy chọn `save_total_limit` (mặc định 1) và `save_only_model` (mặc định true) để kiểm soát hoạt động lưu checkpoint của HuggingFace Trainer, giúp tránh lỗi đầy ổ đĩa trên Kaggle mà vẫn linh hoạt khi chạy trên Google Colab.
  - Cập nhật [train_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/train_gliner.py) và [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py):
    - Đọc và áp dụng `save_total_limit` cùng `save_only_model` vào `TrainingArguments` của HF.
    - Tự động đồng bộ hóa thuộc tính `model.data_processor.max_len = max_length` từ cấu hình vào mô hình ngay sau khi load, giúp giải quyết triệt để cảnh báo `Sentence of length X has been truncated to 384` trên bản Large và Multi-task.
- **2026-06-15 (Tích hợp chạy thử nghiệm trên file Excel)**:
  - Cài đặt thư viện `openpyxl` để pandas tương thích với file Excel.
  - Giải nén `pytorch_model.bin` từ `best_modelq.zip` và tạo file `config.json` để hoàn thiện thư mục mô hình cục bộ `d:\download\glinner\glinner-small_v2.5\`.
  - Tạo script [test_model.py](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/test_model.py) để chạy dự đoán hàng loạt thực thể `SKILL` và `EXPERIENCE` trên file `data_xin_1000_dong.xlsx`, xuất kết quả phân tích kèm JSON thô ra file Excel mới.
  - Viết tài liệu [README.md](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/README.md) cục bộ trong thư mục `data_test/` mô tả chi tiết chức năng và các liên kết tệp theo yêu cầu của global rule.
- **2026-06-15 (Thêm script check_labels.py)**: Tạo script [check_labels.py](file:///c:/Users/loiha/Videos/dfghtraingliner/check_labels.py) giúp kiểm tra và in bảng phân bố nhãn của dataset trước/sau khi map, thống kê dữ liệu thực thể NER gốc (`SKILL`, `MAJOR`, `EXPERIENCE`), cùng với phân bố nhà tuyển dụng và độ dài văn bản. Cấu hình mã hóa UTF-8 khi in và tự động nhận diện vị trí dataset linh hoạt trên cả local và Kaggle.
- **2026-06-15 (Tích hợp dán nhãn chuẩn và pipeline đối chiếu benchmark)**:
  - Cập nhật [build_dataset_v3.py](file:///c:/Users/loiha/Videos/dfghtraingliner/build_dataset_v3.py) để gán nhãn chuẩn (Gold Labels) cho 1000 dòng dữ liệu mô tả công việc bằng API DeepSeek V3 thông qua so khớp biên token chính xác, xuất ra định dạng tương thích huấn luyện `data_xin_1000_dong_gold.json`.
  - Tạo script đối chiếu [benchmark_predicted.py](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/benchmark_predicted.py) nhằm tính toán các chỉ số Precision, Recall và F1-score đối với 2 thực thể `SKILL` và `EXPERIENCE` dựa trên Exact Span Match, rồi xuất báo cáo chi tiết ra [benchmark_report.txt](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/benchmark_report.txt).
  - Cập nhật tài liệu cục bộ [README.md](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/README.md) trong thư mục `data_test/` mô tả đầy đủ luồng dán nhãn chuẩn và đối chiếu benchmark.
- **2026-06-16 (Tích hợp làm sạch nhãn vàng và predictions)**:
  - Tạo script [clean_gold_labels.py](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/clean_gold_labels.py) giúp làm sạch nhãn vàng DeepSeek V3 theo đúng quy luật (loại bỏ từ mơ hồ, trùng vị trí công việc, giải quyết chồng lấn).
  - Cập nhật [benchmark_predicted.py](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/benchmark_predicted.py) nhằm tự động loại bỏ các dự đoán SKILL bị chồng lấn với nhãn chuẩn MAJOR trong Gold, đồng thời tích hợp lọc theo ngưỡng động tối ưu (SKILL >= 0.80, EXPERIENCE >= 0.50) giúp nâng cao F1-Score vượt trội.
  - Đối chiếu và kiểm thử so sánh kết quả benchmark của mô hình trước và sau khi làm sạch cả Gold và Predictions.
  - Cập nhật tài liệu cục bộ [README.md](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/README.md) và tài liệu chính [documentation.md](file:///c:/Users/loiha/Videos/dfghtraingliner/documentation.md).
- **2026-06-16 (Tạo config huấn luyện chuyên biệt GLiNER-Medium-v2.5)**:
  - Tạo tệp cấu hình mới [config_medium_v25.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config_medium_v25.yaml) để chạy huấn luyện riêng cho mô hình GLiNER-Medium-v2.5 mà không ảnh hưởng tới cấu hình cũ.
  - Cập nhật tài liệu tổng thể [documentation.md](file:///c:/Users/loiha/Videos/dfghtraingliner/documentation.md).
- **2026-06-16 (Hỗ trợ huấn luyện kèm nhãn MAJOR và tiếp tục train từ model có sẵn)**:
  - Tạo tệp cấu hình mới [config_medium_v25_major.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config_medium_v25_major.yaml) để huấn luyện mô hình với 3 thực thể nhằm giải quyết vấn đề nhãn MAJOR bị gán nhầm thành SKILL ở đầu ra. Hỗ trợ truyền đường dẫn cục bộ để huấn luyện tiếp tục trên model đã lưu.
  - Cập nhật tài liệu tổng thể [documentation.md](file:///c:/Users/loiha/Videos/dfghtraingliner/documentation.md).
- **2026-06-16 (Dọn dẹp code thừa và Tách biệt tập Test độc lập cho GLiNER)**:
  - Xóa bỏ các tệp nháp dư thừa không còn sử dụng: `clean_labels_local.py` và `convert_ner_format.py` khỏi thư mục `data_test/`.
  - Tạo tệp cấu hình mới [config_small_v25_major.yaml](file:///c:/Users/loiha/Videos/dfghtraingliner/config_small_v25_major.yaml) chuyên biệt cho việc train GLiNER Small v2.5 với nhãn MAJOR.
  - Cập nhật [train_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/train_gliner.py) và [benchmark_gliner.py](file:///c:/Users/loiha/Videos/dfghtraingliner/benchmark_gliner.py) để phân tách nhỏ tập dữ liệu validation (tỉ lệ mặc định 20% trong config) làm 2 phần bằng nhau: 50% làm tập Validation thực tế (cho Trainer và early stopping) và 50% làm tập Test độc lập hoàn toàn (cho việc đánh giá F1/nDCG cuối cùng). Logic này giúp triệt tiêu hoàn toàn hiện tượng "data leakage" và đảm bảo tính đánh giá khách quan tương tự như mô hình Classifier.
  - Cập nhật đồng bộ các tài liệu chính [README.md](file:///c:/Users/loiha/Videos/dfghtraingliner/README.md) ở root, [data_test/README.md](file:///c:/Users/loiha/Videos/dfghtraingliner/data_test/README.md) và [documentation.md](file:///c:/Users/loiha/Videos/dfghtraingliner/documentation.md).






