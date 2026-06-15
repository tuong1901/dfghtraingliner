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
    
    total_samples = len(data)
    
    # 2. General metadata & Text statistics
    providers = Counter()
    lengths = []
    word_counts = []
    
    for item in data:
        providers[item.get("provider", "N/A")] += 1
        text = item.get("text", "")
        lengths.append(len(text))
        word_counts.append(len(text.split()))
        
    print("\n" + "="*60)
    print("  THÔNG TIN TỔNG QUAN VỀ DATASET GỐC (METADATA)")
    print("="*60)
    print(f"  - Tổng số Job Descriptions: {total_samples}")
    print(f"  - Độ dài ký tự: ")
    print(f"      + Trung bình : {sum(lengths)/total_samples:.1f} ký tự")
    print(f"      + Ngắn nhất  : {min(lengths)} ký tự")
    print(f"      + Dài nhất   : {max(lengths)} ký tự")
    print(f"  - Số từ (khoảng trắng): ")
    print(f"      + Trung bình : {sum(word_counts)/total_samples:.1f} từ")
    print(f"      + Ngắn nhất  : {min(word_counts)} từ")
    print(f"      + Dài nhất   : {max(word_counts)} từ")
    print(f"  - Phân bố theo Nhà tuyển dụng / Nguồn (Provider):")
    for prov, count in sorted(providers.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_samples) * 100
        print(f"      + {prov:<10} : {count:<6} ({ratio:.2f}%)")

    # 3. Count NER entities (label field)
    ner_counts = Counter()
    entities_per_jd = []
    for item in data:
        spans = item.get("label", [])
        entities_per_jd.append(len(spans))
        for span in spans:
            if len(span) == 3:
                ner_counts[span[2]] += 1
                
    total_entities = sum(ner_counts.values())
    
    print("\n" + "="*60)
    print("  THỐNG KÊ THỰC THỂ NER GỐC (spans trong trường 'label')")
    print("="*60)
    print(f"  - Tổng số thực thể (spans): {total_entities}")
    print(f"  - Số thực thể trung bình/JD: {total_entities/total_samples:.1f}")
    print(f"  - Phân bố chi tiết các nhãn thực thể:")
    print(f"      {'Nhãn thực thể':<20} | {'Số lượng':<10} | {'Tỉ lệ (%)':<10}")
    print(f"      {'-'*47}")
    for ent, count in sorted(ner_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_entities) * 100
        print(f"      {ent:<20} | {count:<10} | {ratio:.2f}%")
    
    # 4. Count original level labels
    orig_counts = Counter()
    for item in data:
        lvl = str(item.get("level", "")).upper().strip()
        if not lvl:
            orig_counts["[RỖNG/THIẾU]"] += 1
        else:
            orig_counts[lvl] += 1
            
    print("\n" + "="*60)
    print("  BẢNG THỐNG KÊ NHÂN PHÂN LOẠI CẤP BẬC GỐC (ORIGINAL LEVELS)")
    print("="*60)
    print(f"  {'Cấp bậc gốc':<20} | {'Số lượng':<10} | {'Tỉ lệ (%)':<10}")
    print("-" * 60)
    for lvl, count in sorted(orig_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_samples) * 100
        print(f"  {lvl:<20} | {count:<10} | {ratio:.2f}%")
        
    # 5. Count mapped labels
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
    print("  BẢNG THỐNG KÊ NHÃN PHÂN LOẠI CẤP BẬC SAU KHI MAP (TARGET LEVELS)")
    print("="*60)
    print(f"  {'Cấp bậc mới':<20} | {'Số lượng':<10} | {'Tỉ lệ (%)':<10}")
    print("-" * 60)
    for lvl, count in sorted(mapped_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_samples) * 100
        print(f"  {lvl:<20} | {count:<10} | {ratio:.2f}%")

if __name__ == "__main__":
    main()
