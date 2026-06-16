import os
import sys
import json
import re
import shutil
from collections import Counter

def clean_gold_labels():
    # Configure UTF-8 for Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'data_xin_1000_dong_gold.json')
    backup_file = os.path.join(base_dir, 'data_xin_1000_dong_gold_backup.json')
    
    if not os.path.isfile(input_file):
        print(f"❌ Error: {input_file} không tìm thấy!")
        return

    # Back up the original file if backup does not exist yet
    if not os.path.isfile(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"[✓] Đã tạo file sao lưu dữ liệu gốc tại: {backup_file}")
    else:
        print(f"[*] File sao lưu đã tồn tại tại: {backup_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"\n=== BẮT ĐẦU LÀM SẠCH NHÃN VÀNG (GOLD LABELS) TRONG {input_file} ===")
    print(f"Tổng số bản ghi: {len(data)}")
    
    # 1. Blacklist for vague/generic skills
    vague_skills_list = {
        "năng động", "sáng tạo", "cẩn thận", "trung thực", "nhiệt tình", "chăm chỉ",
        "tự giác", "kiên nhẫn", "linh hoạt", "thích nghi", "có trách nhiệm", "trách nhiệm",
        "chịu áp lực", "chịu áp lực cao", "làm thêm giờ", "đạo đức nghề nghiệp",
        "tác phong chuyên nghiệp", "tác phong", "phân tích", "tổng hợp", "báo cáo",
        "quản lý", "phối hợp", "hỗ trợ", "giám sát", "triển khai", "xây dựng",
        "phát triển", "thực hiện", "kỹ năng", "kiến thức", "hiểu biết", "năng lực",
        "khả năng", "chuyên môn", "nghiệp vụ", "quy định ngành", "tiêu chuẩn ngành",
        "quy trình", "quy định", "quy trình nội bộ", "tiêu chuẩn", "làm việc nhóm",
        "làm việc độc lập", "teamwork", "học hỏi nhanh", "ham học hỏi", "cầu tiến",
        "tư duy tốt", "tư duy", "giao tiếp", "giao tiếp tốt", "giao tiếp hiệu quả",
        "thuyết trình", "thuyết phục", "hard working", "fast learner", "team player",
        "self-motivated", "detail oriented", "problem solving", "critical thinking",
        "communication", "time management"
    }
    
    # 2. Job Titles commonly labeled as SKILL
    job_title_pat = re.compile(
        r'\b(developer|engineer|architect|manager|tester|leader|nhân viên|kỹ sư|lập trình viên|chuyên viên|trưởng nhóm)\b',
        re.IGNORECASE
    )
    
    orig_ner = Counter()
    cleaned_ner = Counter()
    
    removed_vague_count = 0
    removed_title_count = 0
    resolved_overlap_count = 0
    total_spans_before = 0
    total_spans_after = 0
    
    for item in data:
        text = item.get('text', '')
        labels = item.get('label', [])
        
        # Step A: Filter out bad labels
        filtered_labels = []
        for start, end, label in labels:
            orig_ner[label] += 1
            total_spans_before += 1
            
            extracted = text[start:end].strip()
            low_extracted = extracted.lower()
            
            if label == 'SKILL' and low_extracted in vague_skills_list:
                removed_vague_count += 1
                continue
                
            if label == 'SKILL' and job_title_pat.search(low_extracted) and low_extracted not in ("project manager", "scrum master"):
                removed_title_count += 1
                continue
                
            filtered_labels.append([start, end, label])
            
        # Step B: Resolve overlapping spans
        # Sort by span length descending to keep the longest/most specific spans
        sorted_by_len = sorted(filtered_labels, key=lambda x: (x[1] - x[0]), reverse=True)
        
        kept_labels = []
        for label_item in sorted_by_len:
            start, end, label_type = label_item
            
            overlap = False
            for k_start, k_end, _ in kept_labels:
                if not (end <= k_start or start >= k_end):
                    overlap = True
                    break
            
            if not overlap:
                kept_labels.append(label_item)
            else:
                resolved_overlap_count += 1
                
        # Sort by start_char ascending
        final_labels = sorted(kept_labels, key=lambda x: x[0])
        item['label'] = final_labels
        
        for start, end, label in final_labels:
            cleaned_ner[label] += 1
            total_spans_after += 1
            
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*75)
    print("=== KẾT QUẢ SO SÁNH NHÃN VÀNG TRƯỚC VÀ SAU KHI LÀM SẠCH ===")
    print("="*75)
    print(f"{'Nhãn':<12} | {'Trước lọc':<11} | {'Sau lọc':<11} | {'Số lượng giảm':<13} | {'Tỉ lệ giảm (%)':<15}")
    print("-" * 75)
    for key in sorted(orig_ner.keys()):
        before = orig_ner[key]
        after = cleaned_ner[key]
        diff = before - after
        pct = (diff / before) * 100 if before > 0 else 0
        print(f"{key:<12} | {before:<11} | {after:<11} | -{diff:<12} | {pct:.2f}%")
    
    print("-" * 75)
    total_diff = total_spans_before - total_spans_after
    total_pct = (total_diff / total_spans_before) * 100
    print(f"{'TỔNG CỘNG':<12} | {total_spans_before:<11} | {total_spans_after:<11} | -{total_diff:<12} | {total_pct:.2f}%")
    
    print(f"\n[+] Đã lưu nhãn sạch đè lên file chuẩn tại: {input_file}")
    print(f"Chi tiết lý do loại bỏ:")
    print(f"  - Loại bỏ do từ chung chung/vague : {removed_vague_count} nhãn")
    print(f"  - Loại bỏ do trùng Job Title      : {removed_title_count} nhãn")
    print(f"  - Loại bỏ do chồng lấn (overlap)  : {resolved_overlap_count} nhãn")

if __name__ == "__main__":
    clean_gold_labels()
