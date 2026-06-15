import os
import sys
import json
import yaml
from collections import Counter
from pathlib import Path

def main():
    # Configure UTF-8 encoding for stdout on Windows/Linux environments
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # 1. Load config to get dataset path and target level labels
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"[Lỗi] Không tìm thấy config.yaml tại {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    dataset_path_cfg = cfg["data"]["dataset_path"]
    dataset_name = Path(dataset_path_cfg).name
    
    # Resolve relative path compared to config.yaml directory
    if not os.path.isabs(dataset_path_cfg):
        dataset_path = (config_path.parent / dataset_path_cfg).resolve()
    else:
        dataset_path = Path(dataset_path_cfg)
        
    if not dataset_path.exists():
        # Fallback 1: check same directory as check_labels.py / config.yaml
        alt_path = config_path.parent / dataset_name
        if alt_path.exists():
            dataset_path = alt_path
        else:
            # Fallback 2: check if dataset is in parent directory
            parent_alt = config_path.parent.parent / dataset_name
            if parent_alt.exists():
                dataset_path = parent_alt
            else:
                print(f"[Lỗi] Không tìm thấy dataset '{dataset_name}'")
                print(f"Đã thử tìm ở:")
                print(f"  - Cấu hình: {(config_path.parent / dataset_path_cfg).resolve()}")
                print(f"  - Cục bộ: {config_path.parent / dataset_name}")
                print(f"  - Thư mục cha: {config_path.parent.parent / dataset_name}")
                return

    print(f"Đang tải dataset từ: {dataset_path} ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Tổng số mẫu: {len(data)}")
    
    # 2. Count original labels
    orig_counts = Counter()
    for item in data:
        # Standardize representation
        lvl = str(item.get("level", "")).upper().strip()
        if not lvl:
            orig_counts["[RỖNG/THIẾU]"] += 1
        else:
            orig_counts[lvl] += 1
            
    print("\n" + "="*60)
    print("  BẢNG THỐNG KÊ NHÃN GỐC TRONG DATASET (ORIGINAL)")
    print("="*60)
    print(f"  {'Cấp bậc gốc':<20} | {'Số lượng':<10} | {'Tỉ lệ (%)':<10}")
    print("-" * 60)
    total_samples = len(data)
    for lvl, count in sorted(orig_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_samples) * 100
        print(f"  {lvl:<20} | {count:<10} | {ratio:.2f}%")
        
    # 3. Count mapped labels
    level_labels = [lv.upper() for lv in cfg["classifier"].get("level_labels", [
        "FRESHER", "JUNIOR", "MIDDLE", "SENIOR", "UNKNOWN"
    ])]
    
    mapped_counts = Counter()
    valid_levels = set(level_labels)
    has_unknown = "UNKNOWN" in valid_levels
    
    for item in data:
        lvl = str(item.get("level", "")).upper().strip()
        if lvl not in valid_levels:
            if has_unknown:
                mapped_counts["UNKNOWN"] += 1
            else:
                mapped_counts["[BỊ LỌC BỎ]"] += 1
        else:
            mapped_counts[lvl] += 1

    print("\n" + "="*60)
    print("  BẢNG THỐNG KÊ NHÃN SAU KHI MAP (TARGET TRAINING)")
    print("="*60)
    print(f"  {'Cấp bậc mới':<20} | {'Số lượng':<10} | {'Tỉ lệ (%)':<10}")
    print("-" * 60)
    for lvl, count in sorted(mapped_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_samples) * 100
        print(f"  {lvl:<20} | {count:<10} | {ratio:.2f}%")

if __name__ == "__main__":
    main()
