import pandas as pd
from datasets import load_dataset

print("Đang tải dữ liệu từ Hugging Face...")
# 1. Tải tập dữ liệu 'train' của tinixai/vietnamese-job-descriptions
dataset = load_dataset("tinixai/vietnamese-job-descriptions", split="train")

# 2. Chuyển thành Pandas DataFrame để xử lý
df = dataset.to_pandas()

# 3. [ORDER BY RANDOM()] Trộn ngẫu nhiên toàn bộ các dòng dữ liệu
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Danh sách các cột bạn sử dụng trong hàm CONCAT_WS
cols_to_concat = [
    'job_title', 
    'experience_level', 
    'job_position', 
    'job_description', 
    'requirements'
]

# Chuyển các cột về dạng chuỗi (string), thay thế các giá trị NaN/Null bằng chuỗi rỗng
for col in cols_to_concat:
    df_shuffled[col] = df_shuffled[col].fillna('').astype(str)

# 5. [CONCAT_WS('  ', ...)] Nối các cột lại với nhau bằng 2 dấu cách
df_shuffled['combined_text'] = df_shuffled[cols_to_concat].agg('  '.join, axis=1)

# 6. [LIMIT 1000] Trích xuất lấy đúng 1000 dòng đầu tiên sau khi trộn
df_result = df_shuffled[['combined_text']].head(1000)

# 7. Xuất kết quả ra file
excel_filename = "vietnamese_jobs_combined_1k.xlsx"
df_result.to_excel(excel_filename, index=False)
print(f"🎉 Hoàn thành! Đã tạo file Excel với 1000 dòng tại: {excel_filename}")

# Nếu bạn muốn dùng file CSV để nạp vào model, mở dòng code phía dưới:
# df_result.to_csv("vietnamese_jobs_combined_1k.csv", index=False, encoding="utf-8-sig")