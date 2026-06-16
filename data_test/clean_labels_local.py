import json
import re
import os

def clean_labels_local():
    # Định nghĩa đường dẫn tuyệt đối theo vị trí thư mục của script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'cleaned_dataset.json')
    output_file = os.path.join(base_dir, 'cleaned_dataset.json')
    
    if not os.path.isfile(input_file):
        print(f"Error: {input_file} not found!")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"=== Running local label cleanup on {input_file} ===")
    print(f"Total input records: {len(data)}")
    
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
    
    cleaned_count = 0
    removed_vague_count = 0
    removed_title_count = 0
    resolved_overlap_count = 0
    
    for item in data:
        text = item.get('text', '')
        labels = item.get('label', [])
        
        # Step A: Filter out bad labels
        filtered_labels = []
        for start, end, label in labels:
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
        cleaned_count += 1
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"\n[+] LOCAL CLEANUP SUCCESSFUL!")
    print(f"    - Removed {removed_vague_count} vague skill labels.")
    print(f"    - Removed {removed_title_count} job title labels.")
    print(f"    - Resolved {resolved_overlap_count} overlapping labels (kept longer spans).")
    print(f"    - Cleaned records saved back to: {output_file}")

if __name__ == "__main__":
    clean_labels_local()
