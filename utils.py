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
    Load cleaned_dataset.json, lọc các sample không hợp lệ,
    rồi chia train/val theo tỉ lệ val_ratio.

    Args:
        dataset_path  : Đường dẫn tới cleaned_dataset.json
        val_ratio     : Tỉ lệ validation (vd: 0.1 = 10%)
        max_samples   : Giới hạn số sample (None = toàn bộ)
        seed          : Random seed để chia nhất quán
        level_labels  : Danh sách các level hợp lệ (để lọc sample có level không hợp lệ)

    Returns:
        (train_data, val_data): Hai list các dict
    """
    print(f"[Data] Đang load dataset từ: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"[Data] Tổng số sample gốc: {len(data)}")
    
    # Lọc sample không có text
    data = [d for d in data if d.get("text", "").strip()]
    print(f"[Data] Sau khi lọc sample không có text: {len(data)}")
    
    # Lọc và map level
    if level_labels is not None:
        valid_levels = set(lv.upper() for lv in level_labels)
        has_unknown = "UNKNOWN" in valid_levels
        
        filtered_data = []
        for d in data:
            d_copy = d.copy()
            lvl = str(d_copy.get("level", "")).upper().strip()
            if lvl not in valid_levels:
                if has_unknown:
                    d_copy["level"] = "UNKNOWN"
                else:
                    continue
            filtered_data.append(d_copy)
        data = filtered_data
        print(f"[Data] Sau khi lọc và map level sang UNKNOWN: {len(data)}")
    
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
