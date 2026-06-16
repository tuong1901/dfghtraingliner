"""
utils.py
--------
Các hàm tiện ích dùng chung cho cả 2 pipeline training:
  - load_config()        : Đọc file config.yaml
  - load_dataset()       : Load và split cleaned_dataset.json
  - set_seed()           : Fix random seed cho reproducibility
  - get_level_label()    : Chuẩn hoá chuỗi level thành label index
  - format_time()        : Format thời gian đẹp khi logging
"""

import os

# Tự động giới hạn chỉ dùng GPU 0 để tránh PyTorch DataParallel (gây tràn VRAM và lỗi NCCL)
# Chỉ áp dụng nếu chạy script đơn lẻ (không chạy DDP/torchrun) và chưa được thiết lập thủ công
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import time
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Tự động cấu hình NCCL để tránh lỗi "NCCL Error 1: unhandled cuda error" trên Kaggle Dual T4 GPU
if "NCCL_P2P_DISABLE" not in os.environ:
    os.environ["NCCL_P2P_DISABLE"] = "1"
if "NCCL_IB_DISABLE" not in os.environ:
    os.environ["NCCL_IB_DISABLE"] = "1"

# Tự động cấu hình bộ nhớ PyTorch CUDA để tránh phân mảnh bộ nhớ (Memory Fragmentation OOM)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------------------------------------------------------
# Đọc config
# ----------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """
    Đọc file config YAML và trả về dict.
    
    Args:
        config_path: Đường dẫn tới config.yaml
    Returns:
        dict chứa toàn bộ config
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ----------------------------------------------------------------
# Fix seed
# ----------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    Fix random seed để đảm bảo reproducibility.
    
    Args:
        seed: Giá trị seed (mặc định 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------
def load_dataset(
    dataset_path: str,
    val_ratio: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42,
    level_labels: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Load cleaned_dataset.json bằng phương pháp stream tiết kiệm bộ nhớ,
    lọc các sample không hợp lệ, rồi chia train/val theo tỉ lệ val_ratio.

    Args:
        dataset_path  : Đường dẫn tới cleaned_dataset.json
        val_ratio     : Tỉ lệ validation (vd: 0.1 = 10%)
        max_samples   : Giới hạn số sample (None = toàn bộ)
        seed          : Random seed để chia nhất quán
        level_labels  : Danh sách các level hợp lệ (để lọc sample có level không hợp lệ)

    Returns:
        (train_data, val_data): Hai list các dict
    """
    print(f"[Data] Đang load dataset (streaming) từ: {dataset_path}")
    
    # Custom streaming JSON array parser để tránh MemoryError khi Ram thấp
    def stream_json_array(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_size = 65536
            found_start = False
            buffer = []
            bracket_count = 0
            in_string = False
            escape = False
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                for char in chunk:
                    if not found_start:
                        if char == '[':
                            found_start = True
                        continue
                    if bracket_count > 0:
                        buffer.append(char)
                    if in_string:
                        if escape:
                            escape = False
                        elif char == '\\':
                            escape = True
                        elif char == '"':
                            in_string = False
                    else:
                        if char == '"':
                            in_string = True
                        elif char == '{':
                            if bracket_count == 0:
                                buffer = [char]
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                obj_str = "".join(buffer)
                                try:
                                    yield json.loads(obj_str)
                                except Exception:
                                    pass
                                buffer = []

    raw_generator = stream_json_array(dataset_path)
    
    # Xử lý lọc và map level on-the-fly để tối ưu bộ nhớ
    valid_levels = set(lv.upper() for lv in level_labels) if level_labels is not None else None
    
    data = []
    for d in raw_generator:
        # Lọc sample không có text
        if not d.get("text", "").strip():
            continue
            
        # Lọc và map level
        if valid_levels is not None:
            lvl = str(d.get("level", "")).upper().strip()
            # Map raw level to target level (Option A)
            if lvl in ["LEAD", "PRINCIPAL", "ARCHITECT", "DIRECTOR"]:
                mapped_lvl = "LEAD_PLUS"
            elif lvl in ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "MANAGER"]:
                mapped_lvl = lvl
            else:
                # Bỏ qua nhãn UNKNOWN thật, EXECUTIVE, hoặc rỗng
                continue
                
            if mapped_lvl in valid_levels:
                d["level"] = mapped_lvl
            else:
                continue
        data.append(d)
        
    print(f"[Data] Tổng số sample hợp lệ sau lọc: {len(data)}")
    
    # Giới hạn số sample
    if max_samples is not None and max_samples < len(data):
        random.seed(seed)
        data = random.sample(data, max_samples)
        print(f"[Data] Đã giới hạn còn: {len(data)} sample")
    
    # Shuffle và chia train/val
    random.seed(seed)
    random.shuffle(data)
    
    n_val = max(1, int(len(data) * val_ratio))
    val_data = data[:n_val]
    train_data = data[n_val:]
    
    print(f"[Data] Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data


# ----------------------------------------------------------------
# Chuẩn hoá level
# ----------------------------------------------------------------
def normalize_level(level_str: str, level_labels: List[str]) -> Optional[int]:
    """
    Chuyển chuỗi level thành index trong level_labels.
    Trả về None nếu không tìm thấy.

    Args:
        level_str   : Giá trị level từ dataset (vd: "SENIOR", "senior")
        level_labels: Danh sách các level hợp lệ theo thứ tự

    Returns:
        index trong level_labels, hoặc None nếu không hợp lệ
    """
    level_upper = str(level_str).upper().strip()
    try:
        return level_labels.index(level_upper)
    except ValueError:
        return None


# ----------------------------------------------------------------
# Format thời gian
# ----------------------------------------------------------------
def format_time(seconds: float) -> str:
    """
    Chuyển số giây thành chuỗi dạng "Xh Ym Zs" để log đẹp.

    Args:
        seconds: Số giây

    Returns:
        Chuỗi thời gian đã format
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


# ----------------------------------------------------------------
# In banner đẹp
# ----------------------------------------------------------------
def print_banner(title: str):
    """In tiêu đề đẹp ra terminal."""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


# ----------------------------------------------------------------
# Kiểm tra GPU
# ----------------------------------------------------------------
def check_device() -> str:
    """
    Kiểm tra xem có GPU không.
    
    Returns:
        "cuda" nếu có GPU, "cpu" nếu không
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] GPU: {gpu_name}")
            return "cuda"
        else:
            print("[Device] Không tìm thấy GPU, dùng CPU")
            return "cpu"
    except ImportError:
        print("[Device] PyTorch chưa cài, dùng CPU")
        return "cpu"
