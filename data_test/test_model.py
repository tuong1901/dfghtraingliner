import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import GLiNER
try:
    from gliner import GLiNER
except ImportError:
    print("[LỖI] Chưa cài đặt thư viện gliner. Vui lòng cài đặt bằng: pip install gliner")
    sys.exit(1)

# Thêm đường dẫn root vào sys.path để import utils nếu cần
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
try:
    from utils import check_device, print_banner, format_time
except ImportError:
    # Fallback nếu không import được từ utils
    def check_device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    def print_banner(title):
        print(f"\n{'='*60}\n  {title}\n{'='*60}\n")
    def format_time(seconds):
        return f"{seconds:.2f}s"

def main():
    print_banner("RUNNING GLiNER NER INFERENCE ON TEST DATASET")

    # 1. Đường dẫn cấu hình
    model_path = r"D:\download\glinner\glinner_main"
    excel_input_path = r"c:\Users\loiha\Videos\dfghtraingliner\data_test\data_xin_1000_dong.xlsx"
    excel_output_path = r"c:\Users\loiha\Videos\dfghtraingliner\data_test\data_xin_1000_dong_predicted.xlsx"
    
    threshold = 0.6
    batch_size = 16

    # 2. Kiểm tra thiết bị phần cứng
    device = check_device()
    print(f"[*] Đang sử dụng thiết bị: {device}")

    # 3. Load model GLiNER
    print(f"[*] Đang tải mô hình GLiNER từ: {model_path}...")
    start_load = time.time()
    try:
        model = GLiNER.from_pretrained(model_path)
        model.to(device)
        print(f"[+] Tải mô hình thành công sau {format_time(time.time() - start_load)}")
    except Exception as e:
        print(f"[LỖI] Không thể tải mô hình: {e}")
        sys.exit(1)

    # 4. Xác định entity types để trích xuất
    entity_types_file = os.path.join(model_path, "entity_types.json")
    if os.path.exists(entity_types_file):
        with open(entity_types_file, "r", encoding="utf-8") as f:
            entity_types = json.load(f).get("entity_types", ["SKILL", "EXPERIENCE"])
    else:
        entity_types = ["SKILL", "EXPERIENCE"]
    print(f"[*] Các nhãn thực thể (entities) cần trích xuất: {entity_types}")

    # 5. Load file Excel kiểm thử
    print(f"[*] Đang đọc dữ liệu đầu vào: {excel_input_path}...")
    if not os.path.exists(excel_input_path):
        print(f"[LỖI] File dữ liệu không tồn tại: {excel_input_path}")
        sys.exit(1)

    try:
        df = pd.read_excel(excel_input_path)
        print(f"[+] Đã đọc thành công {len(df)} dòng dữ liệu.")
    except Exception as e:
        print(f"[LỖI] Không thể đọc file Excel: {e}")
        sys.exit(1)

    # Xác định cột văn bản cần test
    text_col = None
    if "combined_text" in df.columns:
        text_col = "combined_text"
    elif "job_description" in df.columns:
        text_col = "job_description"
    else:
        text_col = df.columns[0]
        print(f"[!] Cột 'combined_text' không tìm thấy, sử dụng cột đầu tiên: '{text_col}'")

    texts = df[text_col].fillna("").astype(str).tolist()

    # 6. Chạy inference dự đoán thực thể
    print("[*] Bắt đầu dự đoán thực thể...")
    all_predictions = []
    start_infer = time.time()

    import torch
    model.eval()
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Đang phân tích text"):
            preds = model.predict_entities(text, entity_types, threshold=threshold)
            all_predictions.append(preds)

    infer_time = time.time() - start_infer
    print(f"[+] Dự đoán hoàn tất sau {format_time(infer_time)} ({len(texts) / infer_time:.2f} mẫu/giây)")

    # 7. Hậu xử lý kết quả dự đoán
    print("[*] Đang tổng hợp và phân loại kết quả...")
    skills_list = []
    experiences_list = []
    raw_json_list = []

    for preds in all_predictions:
        skills = []
        experiences = []
        
        for ent in preds:
            lbl = ent["label"].upper()
            text_val = ent["text"].strip()
            if lbl == "SKILL":
                skills.append(text_val)
            elif lbl == "EXPERIENCE":
                experiences.append(text_val)

        # Lọc trùng và sắp xếp
        unique_skills = sorted(list(set(skills)))
        unique_experiences = sorted(list(set(experiences)))

        skills_list.append(", ".join(unique_skills) if unique_skills else "N/A")
        experiences_list.append(", ".join(unique_experiences) if unique_experiences else "N/A")
        raw_json_list.append(json.dumps(preds, ensure_ascii=False))

    # Ghi kết quả vào DataFrame mới
    df["predicted_skills"] = skills_list
    df["predicted_experience"] = experiences_list
    df["predicted_entities_raw_json"] = raw_json_list

    # 8. Lưu kết quả ra file Excel mới
    print(f"[*] Đang xuất kết quả ra file: {excel_output_path}...")
    try:
        df.to_excel(excel_output_path, index=False)
        print(f"[+] Hoàn thành! Kết quả dự đoán được lưu tại: {excel_output_path}")
    except Exception as e:
        print(f"[LỖI] Không thể ghi kết quả ra file Excel: {e}")
        sys.exit(1)

    # 9. Thống kê kết quả
    total_skills = sum([1 for s in skills_list if s != "N/A"])
    total_exp = sum([1 for e in experiences_list if e != "N/A"])
    print("\n" + "="*40 + " THỐNG KÊ KẾT QUẢ " + "="*40)
    print(f"  - Tổng số mẫu đã test: {len(df)}")
    print(f"  - Số tin tuyển dụng có SKILL: {total_skills} ({total_skills/len(df)*100:.2f}%)")
    print(f"  - Số tin tuyển dụng có EXPERIENCE: {total_exp} ({total_exp/len(df)*100:.2f}%)")
    print(f"  - Tốc độ xử lý trung bình: {len(df)/infer_time:.2f} câu/giây")
    print("="*98)

if __name__ == "__main__":
    main()
