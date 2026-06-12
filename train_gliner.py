"""
train_gliner.py
---------------
Script train GLiNER model cho bài toán NER (SKILL, MAJOR, EXPERIENCE).

Sử dụng:
    python train_gliner.py [--config config.yaml]

GLiNER là model NER zero/few-shot dựa trên bi-encoder, finetune bằng
cách truyền vào danh sách entity types dạng text + spans ký tự (char-level).

Dataset format đầu vào (từ cleaned_dataset.json):
    {
        "text": "...",
        "label": [[start, end, "SKILL"], [start, end, "MAJOR"], ...],
        "level": "SENIOR"
    }

Output:
    Model được lưu vào output_dir trong config (mặc định: ./outputs/gliner)

Hàm chính:
    - prepare_gliner_samples()  : Chuyển dataset sang format GLiNER
    - train_gliner()            : Train loop chính
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Thêm thư mục cha vào sys.path để import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, load_dataset, set_seed,
    print_banner, check_device, format_time
)


# ----------------------------------------------------------------
# Chuẩn bị dữ liệu cho GLiNER
# ----------------------------------------------------------------
def prepare_gliner_samples(
    data: List[dict],
    entity_types: List[str],
    max_length: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Chuyển đổi dữ liệu từ cleaned_dataset.json sang format mà
    GLiNER trainer yêu cầu.

    GLiNER format:
    {
        "tokenized_text": [...],  # Sẽ được tokenize trong trainer
        "ner": [[start_char, end_char, "LABEL"], ...]
    }

    Thực ra GLiNER dùng format:
    {
        "text": "raw text...",
        "entities": [{"start": int, "end": int, "label": str}, ...]
    }

    Args:
        data        : List các dict từ dataset
        entity_types: Các loại entity được train (filter các label khác)
        max_length  : Giới hạn độ dài text (ký tự) để tránh OOM

    Returns:
        List các dict theo GLiNER format
    """
    valid_types = set(t.upper() for t in entity_types)
    samples = []
    skipped = 0
    
    for item in data:
        text = item.get("text", "").strip()
        labels = item.get("label", [])
        
        if not text:
            skipped += 1
            continue
        
        # Cắt text nếu quá dài (theo ký tự, xấp xỉ)
        # GLiNER tokenizer sẽ xử lý chính xác hơn
        if len(text) > max_length * 4:  # Ước lượng ~4 ký tự/token
            text = text[:max_length * 4]
        
        # Lọc entities hợp lệ
        entities = []
        for span in labels:
            if len(span) != 3:
                continue
            start, end, label = span
            label_upper = str(label).upper()
            
            # Chỉ lấy các label trong entity_types
            if label_upper not in valid_types:
                continue
            
            # Kiểm tra bounds
            if start < 0 or end > len(text) or start >= end:
                continue
            
            # Kiểm tra text tại span không rỗng
            span_text = text[start:end].strip()
            if not span_text:
                continue
            
            entities.append({
                "start": start,
                "end": end,
                "label": label_upper
            })
        
        samples.append({
            "text": text,
            "entities": entities,
            "ner": [[ent["start"], ent["end"], ent["label"]] for ent in entities]
        })
    
    if skipped > 0:
        print(f"[GLiNER] Bỏ qua {skipped} sample không hợp lệ")
    
    return samples


# ----------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------
# Helper function to dynamically resolve DataCollator in GLiNER v0.1.x vs v0.2.x
def get_gliner_data_collator(model):
    """
    Tạo data collator tương thích động với phiên bản GLiNER được cài đặt.
    """
    try:
        from gliner.data_processing.collator import DataCollator
        return DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    except ImportError:
        import gliner.data_processing.collator as collator_mod
        processor_class_name = model.data_processor.__class__.__name__
        if "Span" in processor_class_name:
            for class_name in ["SpanDataCollator", "BiEncoderSpanDataCollator"]:
                if hasattr(collator_mod, class_name):
                    return getattr(collator_mod, class_name)(model.config, data_processor=model.data_processor, prepare_labels=True)
        elif "Token" in processor_class_name:
            for class_name in ["TokenDataCollator", "BiEncoderTokenDataCollator"]:
                if hasattr(collator_mod, class_name):
                    return getattr(collator_mod, class_name)(model.config, data_processor=model.data_processor, prepare_labels=True)
        
        for class_name in ["SpanDataCollator", "BiEncoderSpanDataCollator", "DataCollator"]:
            if hasattr(collator_mod, class_name):
                return getattr(collator_mod, class_name)(model.config, data_processor=model.data_processor, prepare_labels=True)
        raise ImportError("Không tìm thấy class DataCollator phù hợp trong gliner.data_processing.collator")


def train_gliner(cfg: dict):
    """
    Train GLiNER model dựa trên config.

    Pipeline:
    1. Load & chuẩn bị dataset
    2. Khởi tạo GLiNER model từ pretrained
    3. Setup Trainer với TrainingArguments
    4. Train & evaluate
    5. Save model và tokenizer

    Args:
        cfg: Dict config đã load từ config.yaml
    """
    print_banner("TRAINING GLiNER NER MODEL")
    
    # --- Import các thư viện cần thiết ---
    try:
        from gliner import GLiNER
        from gliner.training import Trainer, TrainingArguments
    except ImportError as e:
        print(f"\n[LỖI] Chưa cài gliner hoặc lỗi import: {e}")
        print("Chạy lệnh sau:")
        print("  pip install gliner")
        sys.exit(1)
    
    gcfg = cfg["gliner"]
    data_cfg = cfg["data"]
    seed = cfg["run"].get("seed", 42)
    
    set_seed(seed)
    device = check_device()
    
    # 1. Load dataset
    train_data, val_data = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        val_ratio=data_cfg.get("val_ratio", 0.1),
        max_samples=data_cfg.get("max_samples", None),
        seed=seed,
    )
    
    # 2. Chuẩn bị samples
    entity_types = gcfg.get("entity_types", ["SKILL", "MAJOR", "EXPERIENCE"])
    max_length = gcfg.get("max_length", 1024)
    
    print(f"[GLiNER] Entity types: {entity_types}")
    print(f"[GLiNER] Chuẩn bị train samples...")
    train_samples = prepare_gliner_samples(train_data, entity_types, max_length)
    val_samples = prepare_gliner_samples(val_data, entity_types, max_length)
    
    print(f"[GLiNER] Train: {len(train_samples)} | Val: {len(val_samples)}")
    
    # 3. Load model
    model_name = gcfg.get("model_name", "urchade/gliner_small-v2.1")
    print(f"\n[GLiNER] Load model: {model_name}")
    model = GLiNER.from_pretrained(model_name)
    
    # 4. Training arguments
    output_dir = gcfg.get("output_dir", "./outputs/gliner")
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile model nếu có thể (PyTorch 2.0+)
    try:
        import torch
        if hasattr(torch, "compile"):
            print("[GLiNER] Đang compile model (torch.compile)...")
            model = torch.compile(model)
    except Exception:
        pass
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=gcfg.get("learning_rate", 5e-5),
        weight_decay=gcfg.get("weight_decay", 0.01),
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=gcfg.get("warmup_ratio", 0.1),
        per_device_train_batch_size=gcfg.get("train_batch_size", 8),
        per_device_eval_batch_size=gcfg.get("eval_batch_size", 8),
        num_train_epochs=gcfg.get("num_epochs", 5),
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=gcfg.get("save_steps", 500),
        eval_steps=gcfg.get("eval_steps", 500),
        logging_steps=gcfg.get("logging_steps", 50),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if gcfg.get("wandb_project") else "none",
        seed=seed,
        dataloader_num_workers=0,
        fp16=device == "cuda",          # Dùng mixed precision nếu có GPU
        gradient_accumulation_steps=2,  # Tích luỹ gradient để tăng effective batch size
    )
    
    # Setup wandb nếu cần
    if gcfg.get("wandb_project"):
        try:
            import wandb
            wandb.init(project=gcfg["wandb_project"], config=gcfg)
        except ImportError:
            print("[CẢNH BÁO] wandb chưa cài, bỏ qua wandb logging")
    
    # 5. Train
    print("\n[GLiNER] Bắt đầu training...")
    start_time = time.time()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_samples,
        eval_dataset=val_samples,
        data_collator=get_gliner_data_collator(model),
    )
    
    trainer.train()
    
    elapsed = format_time(time.time() - start_time)
    print(f"\n[GLiNER] Training hoàn tất sau {elapsed}")
    
    # 6. Save model
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    print(f"[GLiNER] Model đã lưu tại: {final_dir}")
    
    # 7. Lưu entity_types vào model config để tiện dùng sau
    config_path = os.path.join(final_dir, "entity_types.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"entity_types": entity_types}, f, ensure_ascii=False, indent=2)
    print(f"[GLiNER] Entity types config: {config_path}")
    
    return final_dir


# ----------------------------------------------------------------
# Quick inference test sau khi train
# ----------------------------------------------------------------
def quick_test_gliner(model_dir: str, entity_types: List[str]):
    """
    Test nhanh model GLiNER vừa train với 1 câu mẫu.

    Args:
        model_dir   : Thư mục chứa model đã train
        entity_types: Danh sách entity type
    """
    print_banner("QUICK TEST GLiNER")
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained(model_dir)
        
        test_text = (
            "We are looking for a Senior Python Developer "
            "with 3+ years of experience in Django and FastAPI. "
            "Bachelor's degree in Computer Science preferred."
        )
        
        print(f"Text: {test_text}\n")
        entities = model.predict_entities(test_text, entity_types, threshold=0.5)
        
        if entities:
            for ent in entities:
                print(f"  [{ent['label']}] '{ent['text']}' ({ent['start']}-{ent['end']}, score={ent['score']:.3f})")
        else:
            print("  Không phát hiện entity nào (threshold=0.5)")
    except Exception as e:
        print(f"[CẢNH BÁO] Quick test thất bại: {e}")


# ----------------------------------------------------------------
# Entry point độc lập (dùng khi gọi trực tiếp)
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER NER Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Đường dẫn config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    final_dir = train_gliner(cfg)
    
    entity_types = cfg["gliner"].get("entity_types", ["SKILL", "MAJOR", "EXPERIENCE"])
    quick_test_gliner(final_dir, entity_types)
