"""
build_dataset_v3.py — NER annotation pipeline dùng DeepSeek V3.

Đọc dữ liệu từ file Excel data_test/data_xin_1000_dong.xlsx, gửi đến DeepSeek V3 để
dán nhãn thực thể chuẩn (gold labels) cho 3 loại nhãn: MAJOR, SKILL, EXPERIENCE.
Output: data_test/data_xin_1000_dong_gold.json

Setup:
  1. Thiết lập biến môi trường DEEPSEEK_API_KEY hoặc tạo file .env trong thư mục hiện tại.
  2. Chạy: python data_test/build_dataset_v3.py (hoặc từ thư mục gốc)
"""

import os
import sys
import json
import asyncio
import re
import warnings
import time
import pandas as pd
from collections import Counter, defaultdict

from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIStatusError
import httpx

sys.stdout.reconfigure(encoding='utf-8')

# ── Load .env ─────────────────────────────────────────────────────────────────
_base_dir     = os.path.dirname(os.path.abspath(__file__))
# Load .env từ thư mục hiện tại
load_dotenv(dotenv_path=os.path.join(_base_dir, '.env'))

# ── DeepSeek config ───────────────────────────────────────────────────────────
# Đọc key từ biến môi trường (sau khi load .env)
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('sk-299fa7a5f8034c86a4c273074d65ab37', '').strip()
if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith('sk-...'):
    print("❌ Thiếu DEEPSEEK_API_KEY hoặc key không hợp lệ!")
    print("   → Vui lòng thiết lập biến môi trường DEEPSEEK_API_KEY hoặc tạo file .env chứa: DEEPSEEK_API_KEY=sk-...")
    sys.exit(1)

MODEL       = os.getenv('MODEL', 'deepseek-chat')
CONCURRENCY = int(os.getenv('CONCURRENCY_LIMIT', '15'))  # Concurrency limit để tránh chạm rate limit TPM/RPM

print(f"[*] Provider  : DeepSeek")
print(f"[*] Model     : {MODEL}")
print(f"[*] Concurrency: {CONCURRENCY}")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_EXCEL_FILE = os.path.join(_base_dir, 'data_test', 'data_xin_1000_dong.xlsx')
OUTPUT_FILE      = os.path.join(_base_dir, 'data_test', 'data_xin_1000_dong_gold.json')
PROGRESS_FILE    = OUTPUT_FILE.replace('.json', '_progress.jsonl')  # resume support

print(f"[*] Input Excel : {INPUT_EXCEL_FILE}")
print(f"[*] Output Gold : {OUTPUT_FILE}")

LABELS = ["MAJOR", "SKILL", "EXPERIENCE"]

# ── SSL bypass ────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
_orig = httpx.AsyncClient.__init__
def _no_verify(self, *a, **kw): kw['verify'] = False; _orig(self, *a, **kw)
httpx.AsyncClient.__init__ = _no_verify
os.environ["CURL_CA_BUNDLE"] = os.environ["REQUESTS_CA_BUNDLE"] = ""

# ── DeepSeek client ───────────────────────────────────────────────────────────
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url='https://api.deepseek.com',
)

semaphore = asyncio.Semaphore(CONCURRENCY)

# ── Tokenizer ────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[\w']+|[.,!?;()&]")

def tokenize_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)

def find_token_indices(full_tokens: list[str], ent_tokens: list[str]) -> tuple[int, int]:
    n = len(ent_tokens)
    for i in range(len(full_tokens) - n + 1):
        if full_tokens[i:i + n] == ent_tokens:
            return i, i + n - 1
    return -1, -1


# ── Post-processing: blacklist entity rác ────────────────────────────────────
_BLACKLIST: set[str] = {
    # Personality traits — NOT skills
    "năng động","sáng tạo","cẩn thận","trung thực","nhiệt tình","chăm chỉ",
    "tự giác","kiên nhẫn","linh hoạt","thích nghi","có trách nhiệm","trách nhiệm",
    "chịu áp lực","chịu áp lực cao","làm thêm giờ","đạo đức nghề nghiệp",
    "tác phong chuyên nghiệp","tác phong",
    # Generic actions alone
    "phân tích","tổng hợp","báo cáo","quản lý","phối hợp","hỗ trợ",
    "giám sát","triển khai","xây dựng","phát triển","thực hiện",
    # Vague phrases
    "kỹ năng","kiến thức","hiểu biết","năng lực","khả năng","chuyên môn",
    "nghiệp vụ","quy định ngành","tiêu chuẩn ngành","quy trình","quy định",
    "quy trình nội bộ","tiêu chuẩn","làm việc nhóm","làm việc độc lập","teamwork",
    "học hỏi nhanh","ham học hỏi","cầu tiến","tư duy tốt","tư duy",
    "giao tiếp","giao tiếp tốt","giao tiếp hiệu quả","thuyết trình","thuyết phục",
    # English vague
    "hard working","fast learner","team player","self-motivated","detail oriented",
    "problem solving","critical thinking","communication","time management",
}

_EXP_NUM_RE = re.compile(
    r'\d|fresh\s*graduate|không\s*yêu\s*cầu|chưa\s*có|dưới\s*1|vừa\s*tốt\s*nghiệp|no\s*experience',
    re.IGNORECASE | re.UNICODE,
)

def is_valid_entity(text: str, label: str) -> bool:
    t = text.strip()
    low = t.lower()
    if len(t) < 2:
        return False
    if len(tokenize_text(t)) > 8:   # boundary quá rộng
        return False
    if label == 'SKILL':
        if low in _BLACKLIST:
            return False
        bad_prefixes = ("có trách","có khả năng","có kinh nghiệm về",
                        "am hiểu quy","nắm vững quy","hiểu biết về")
        if any(low.startswith(p) for p in bad_prefixes):
            return False
    if label == 'EXPERIENCE':
        if not _EXP_NUM_RE.search(t):
            return False
    return True


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are a STRICT, precision-first NER annotator for job descriptions (JDs) in Vietnamese and English.
Task: extract entities of exactly 3 types: SKILL, MAJOR, EXPERIENCE.

=== DEFINITIONS ===

SKILL — A specific, nameable, look-up-able technical concept:
  VALID: programming languages (Python, Java, C++), frameworks (React, Spring Boot),
    databases (PostgreSQL, Redis), cloud (AWS, Azure, GCP), tools (Docker, Git, Jira),
    certifications (ISTQB, PMP, ACCA, CFA, TOEIC 700+), methodologies (Agile, Scrum, CI/CD),
    foreign languages (Tiếng Anh, Tiếng Hàn), domain-specific NAMED concepts
    (Machine Learning, IFRS, Core Banking, Luật Lao động, Hải quan),
    domain-specific skills WITH clear context (giao tiếp bệnh nhân, tư vấn thuốc,
    phân tích báo cáo tài chính, quản lý rủi ro tín dụng).

  NOT SKILL — DO NOT label these:
    • Personality: năng động, cẩn thận, nhiệt tình, chịu áp lực cao, có trách nhiệm
    • Vague verbs alone: phân tích, tổng hợp, báo cáo, quản lý, phối hợp
    • Industry jargon: quy định ngành, tiêu chuẩn nghề nghiệp, nghiệp vụ, quy trình nội bộ
    • Generic teamwork: làm việc nhóm, làm việc độc lập, teamwork
    • Generic communication: giao tiếp (alone), giao tiếp tốt (alone)
    • Generic learning: học hỏi nhanh, ham học hỏi, cầu tiến

MAJOR — Academic field of study ONLY:
  VALID: Computer Science, Kế toán, Tài chính - Ngân hàng, Công nghệ thông tin, Luật
  NOT MAJOR: job titles, certifications, company names

EXPERIENCE — Work duration with a number or keyword:
  VALID: "3+ years", "Tối thiểu 3 năm kinh nghiệm", "Fresh graduate", "Không yêu cầu"
  NOT EXPERIENCE: "có kinh nghiệm" (no number), project durations, responsibilities

=== RULES ===
1. VERBATIM: copy entity text EXACTLY as in the JD.
2. SHORTEST SPAN: label the most precise span.
3. SEPARATE: "Java, Python, Go" → 3 entities.
4. EXHAUSTIVE for real skills (don't miss tools/techs/certs).
5. STRICT on vague phrases: if unsure → DO NOT label.
6. Output JSON: {"entities": [{"text": "...", "label": "SKILL|MAJOR|EXPERIENCE"}]}
"""

# ── Few-shot examples (chat format) ──────────────────────────────────────────
_FEW_SHOT = [
    # Example 1: Embedded
    {"role": "user", "content": (
        '### EXAMPLE 1 — Embedded\n'
        'TEXT: "Develop AUTOSAR SWC/BSW in C/C++. Required: CAN, LIN, FlexRay, '
        'ECU integration. Tools: Vector DaVinci. Standards: MISRA C, ASPICE, ISO 26262. '
        'Dev: Git, Jenkins, CI/CD, Agile/Scrum. 3+ years. B.S. Computer Science."\nOUTPUT:'
    )},
    {"role": "assistant", "content": json.dumps({"entities": [
        {"text": "AUTOSAR",        "label": "SKILL"},
        {"text": "SWC",            "label": "SKILL"},
        {"text": "BSW",            "label": "SKILL"},
        {"text": "C/C++",          "label": "SKILL"},
        {"text": "CAN",            "label": "SKILL"},
        {"text": "LIN",            "label": "SKILL"},
        {"text": "FlexRay",        "label": "SKILL"},
        {"text": "ECU integration","label": "SKILL"},
        {"text": "Vector DaVinci", "label": "SKILL"},
        {"text": "MISRA C",        "label": "SKILL"},
        {"text": "ASPICE",         "label": "SKILL"},
        {"text": "ISO 26262",      "label": "SKILL"},
        {"text": "Git",            "label": "SKILL"},
        {"text": "Jenkins",        "label": "SKILL"},
        {"text": "CI/CD",          "label": "SKILL"},
        {"text": "Agile/Scrum",    "label": "SKILL"},
        {"text": "3+ years",       "label": "EXPERIENCE"},
        {"text": "Computer Science","label": "MAJOR"},
    ]}, ensure_ascii=False)},

    # Example 2: NEGATIVE — show what NOT to label
    {"role": "user", "content": (
        '### EXAMPLE 2 — Sales (NEGATIVE: many things NOT labeled)\n'
        'TEXT: "Nhân viên Kinh doanh - 1 năm kinh nghiệm. Tốt nghiệp Quản trị kinh doanh '
        'hoặc Marketing. Năng động, nhiệt tình, chịu áp lực cao, làm việc nhóm tốt. '
        'Thành thạo Excel, PowerPoint, CRM. Kỹ năng giao tiếp tốt. Salesforce là lợi thế."\nOUTPUT:'
    )},
    {"role": "assistant", "content": json.dumps({"entities": [
        {"text": "1 năm kinh nghiệm",   "label": "EXPERIENCE"},
        {"text": "Quản trị kinh doanh", "label": "MAJOR"},
        {"text": "Marketing",           "label": "MAJOR"},
        {"text": "Excel",               "label": "SKILL"},
        {"text": "PowerPoint",          "label": "SKILL"},
        {"text": "CRM",                 "label": "SKILL"},
        {"text": "Salesforce",          "label": "SKILL"},
    ]}, ensure_ascii=False)},

    # Example 3: Backend/Cloud
    {"role": "user", "content": (
        '### EXAMPLE 3 — Backend / Cloud\n'
        'TEXT: "Senior Backend Engineer, 5+ years. Stack: Golang, Python, gRPC, REST API, '
        'GraphQL, Docker, Kubernetes, AWS, GCP, Terraform, PostgreSQL, Redis, Kafka, '
        'Datadog, Git, TDD, Agile. English. B.S. Computer Science."\nOUTPUT:'
    )},
    {"role": "assistant", "content": json.dumps({"entities": [
        {"text": "Golang",          "label": "SKILL"},
        {"text": "Python",          "label": "SKILL"},
        {"text": "gRPC",            "label": "SKILL"},
        {"text": "REST API",        "label": "SKILL"},
        {"text": "GraphQL",         "label": "SKILL"},
        {"text": "Docker",          "label": "SKILL"},
        {"text": "Kubernetes",      "label": "SKILL"},
        {"text": "AWS",             "label": "SKILL"},
        {"text": "GCP",             "label": "SKILL"},
        {"text": "Terraform",       "label": "SKILL"},
        {"text": "PostgreSQL",      "label": "SKILL"},
        {"text": "Redis",           "label": "SKILL"},
        {"text": "Kafka",           "label": "SKILL"},
        {"text": "Datadog",         "label": "SKILL"},
        {"text": "Git",             "label": "SKILL"},
        {"text": "TDD",             "label": "SKILL"},
        {"text": "Agile",           "label": "SKILL"},
        {"text": "English",         "label": "SKILL"},
        {"text": "5+ years",        "label": "EXPERIENCE"},
        {"text": "Computer Science","label": "MAJOR"},
    ]}, ensure_ascii=False)},

    # Example 4: Finance Vietnamese
    {"role": "user", "content": (
        '### EXAMPLE 4 — Finance / Accounting (Vietnamese)\n'
        'TEXT: "Tốt nghiệp Tài chính - Ngân hàng, Kế toán hoặc Kinh tế. '
        'Tối thiểu 3 năm kinh nghiệm. Excel, SQL, SAP, IFRS. CPA, ACCA, CFA là lợi thế. '
        'Tiếng Anh TOEIC 700+. Kỹ năng phân tích báo cáo tài chính."\nOUTPUT:'
    )},
    {"role": "assistant", "content": json.dumps({"entities": [
        {"text": "Tài chính - Ngân hàng",        "label": "MAJOR"},
        {"text": "Kế toán",                      "label": "MAJOR"},
        {"text": "Kinh tế",                      "label": "MAJOR"},
        {"text": "3 năm kinh nghiệm",            "label": "EXPERIENCE"},
        {"text": "Excel",                        "label": "SKILL"},
        {"text": "SQL",                          "label": "SKILL"},
        {"text": "SAP",                          "label": "SKILL"},
        {"text": "IFRS",                         "label": "SKILL"},
        {"text": "CPA",                          "label": "SKILL"},
        {"text": "ACCA",                         "label": "SKILL"},
        {"text": "CFA",                          "label": "SKILL"},
        {"text": "Tiếng Anh",                    "label": "SKILL"},
        {"text": "TOEIC 700+",                   "label": "SKILL"},
        {"text": "phân tích báo cáo tài chính",  "label": "SKILL"},
    ]}, ensure_ascii=False)},

    # Example 5: Data Engineer
    {"role": "user", "content": (
        '### EXAMPLE 5 — Data Engineer (REQUIREMENTS section with certs)\n'
        'TEXT: "REQUIREMENTS: Bachelor in Computer Science, Information Technology or Data Science. '
        '3-5 years as Data Engineer. SQL/Spark SQL, Python & PySpark, ETL/ELT, data modeling, '
        'Azure Databricks, Data Factory, ADLS Gen2, Delta Lake. '
        'Microsoft Azure Data Engineer Associate (DP-203) preferred. '
        'Azure Fundamentals (AZ-900) is a plus. Knowledge of GDPR, HIPAA, RBAC/ABAC."\nOUTPUT:'
    )},
    {"role": "assistant", "content": json.dumps({"entities": [
        {"text": "Computer Science",                              "label": "MAJOR"},
        {"text": "Information Technology",                       "label": "MAJOR"},
        {"text": "Data Science",                                 "label": "MAJOR"},
        {"text": "3-5 years",                                    "label": "EXPERIENCE"},
        {"text": "SQL",                                          "label": "SKILL"},
        {"text": "Spark SQL",                                    "label": "SKILL"},
        {"text": "Python",                                       "label": "SKILL"},
        {"text": "PySpark",                                      "label": "SKILL"},
        {"text": "ETL/ELT",                                      "label": "SKILL"},
        {"text": "data modeling",                                "label": "SKILL"},
        {"text": "Azure Databricks",                             "label": "SKILL"},
        {"text": "Data Factory",                                 "label": "SKILL"},
        {"text": "ADLS Gen2",                                    "label": "SKILL"},
        {"text": "Delta Lake",                                   "label": "SKILL"},
        {"text": "Microsoft Azure Data Engineer Associate (DP-203)", "label": "SKILL"},
        {"text": "Azure Fundamentals (AZ-900)",                  "label": "SKILL"},
        {"text": "GDPR",                                         "label": "SKILL"},
        {"text": "HIPAA",                                        "label": "SKILL"},
        {"text": "RBAC/ABAC",                                    "label": "SKILL"},
    ]}, ensure_ascii=False)},

    # Example 6: Medical Vietnamese
    {"role": "user", "content": (
        '### EXAMPLE 6 — Medical (Vietnamese)\n'
        'TEXT: "Tuyển Dược sĩ lâm sàng. Bằng Dược sĩ đại học trở lên. '
        'Kinh nghiệm 3 năm tại bệnh viện. Am hiểu Dược lý, GPP, GMP. '
        'Kỹ năng tư vấn thuốc, giao tiếp bệnh nhân. Tiếng Anh chuyên ngành. '
        'Năng động, nhiệt tình."\nOUTPUT:'
    )},
    {"role": "assistant", "content": json.dumps({"entities": [
        {"text": "Dược sĩ đại học",        "label": "MAJOR"},
        {"text": "Kinh nghiệm 3 năm",      "label": "EXPERIENCE"},
        {"text": "Dược lý",                "label": "SKILL"},
        {"text": "GPP",                    "label": "SKILL"},
        {"text": "GMP",                    "label": "SKILL"},
        {"text": "tư vấn thuốc",           "label": "SKILL"},
        {"text": "giao tiếp bệnh nhân",    "label": "SKILL"},
        {"text": "Tiếng Anh chuyên ngành", "label": "SKILL"},
    ]}, ensure_ascii=False)},
]


# ── LLM call với retry ────────────────────────────────────────────────────────
async def extract_entities_from_text(text: str, retries: int = 3) -> list[dict]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        *_FEW_SHOT,
        {"role": "user", "content": (
            "### NOW ANNOTATE — scan the ENTIRE text, do NOT stop until all real entities found\n"
            f'TEXT: "{text}"\nOUTPUT:'
        )},
    ]

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096,
            )
            raw  = resp.choices[0].message.content
            data = json.loads(raw)
            if "entities" in data and isinstance(data["entities"], list):
                return data["entities"]
            for v in data.values():
                if isinstance(v, list):
                    return v
            return []

        except RateLimitError:
            wait = 2 ** attempt * 10
            print(f"\n  ⏳ Rate limit hit — chờ {wait}s (attempt {attempt+1}/{retries})")
            await asyncio.sleep(wait)

        except APIStatusError as e:
            if e.status_code == 429:
                wait = 2 ** attempt * 10
                print(f"\n  ⏳ 429 Too Many Requests — chờ {wait}s")
                await asyncio.sleep(wait)
            elif e.status_code == 503:
                wait = 2 ** attempt * 15
                print(f"\n  ⏳ 503 Service Unavailable — chờ {wait}s")
                await asyncio.sleep(wait)
            else:
                print(f"\n  ⚠️  API error {e.status_code}: {e.message}")
                return []

        except Exception as e:
            print(f"\n  ⚠️  Error: {e}")
            return []

    return []


async def process_job(job_text: str, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        if not job_text or len(job_text.strip()) < 10:
            return None

        llm_entities = await extract_entities_from_text(job_text)
        if llm_entities is None:
            return None

        # Tokenize và lấy offsets
        tokens = []
        starts = []
        ends = []
        for match in _TOKEN_RE.finditer(job_text):
            tokens.append(match.group())
            starts.append(match.start())
            ends.append(match.end())

        ner: list[list] = []
        char_labels: list[list] = []

        for ent in llm_entities:
            ent_text = ent.get('text', '').strip()
            label    = ent.get('label', '').upper()

            if label not in LABELS or not ent_text:
                continue
            if not is_valid_entity(ent_text, label):
                continue

            ent_tokens = tokenize_text(ent_text)
            if not ent_tokens:
                continue

            si, ei = find_token_indices(tokens, ent_tokens)
            if si != -1:
                ner.append([si, ei, label])
                char_labels.append([starts[si], ends[ei], label])

        return {
            "text": job_text,
            "label": char_labels,
            "tokenized_text": tokens,
            "ner": ner
        }


# ── Progress / Resume ─────────────────────────────────────────────────────────
def load_progress() -> set[int]:
    done: set[int] = set()
    if os.path.isfile(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec['idx'])
                except Exception:
                    pass
        print(f"[*] Resume: {len(done)} records đã xử lý trước đó")
    return done


def save_progress_record(idx: int, sample: dict | None):
    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'idx': idx, 'sample': sample}, ensure_ascii=False) + '\n')





def load_progress_samples() -> list[dict]:
    idx_to_sample = {}
    if not os.path.isfile(PROGRESS_FILE):
        return []
    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get('sample') is not None:
                    idx_to_sample[int(rec['idx'])] = rec['sample']
            except Exception:
                pass
    
    # Sắp xếp theo thứ tự idx gốc để đồng bộ với Excel
    sorted_samples = [idx_to_sample[idx] for idx in sorted(idx_to_sample.keys())]
    return sorted_samples


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # Đọc input Excel
    if not os.path.isfile(INPUT_EXCEL_FILE):
        print(f"❌ Không tìm thấy input Excel: {INPUT_EXCEL_FILE}")
        sys.exit(1)

    try:
        df = pd.read_excel(INPUT_EXCEL_FILE)
        print(f"[*] Đọc file Excel thành công: {INPUT_EXCEL_FILE}")
    except Exception as e:
        print(f"❌ Lỗi đọc file Excel: {e}")
        sys.exit(1)

    # Xác định cột text
    text_col = None
    if "combined_text" in df.columns:
        text_col = "combined_text"
    elif "job_description" in df.columns:
        text_col = "job_description"
    else:
        text_col = df.columns[0]
    
    print(f"[*] Sử dụng cột dữ liệu văn bản: '{text_col}'")
    texts = df[text_col].fillna("").astype(str).tolist()
    print(f"[*] Tổng số bản ghi cần xử lý: {len(texts):,}")

    # Resume: bỏ qua records đã làm
    done_indices = load_progress()
    todo = [(i, text) for i, text in enumerate(texts) if i not in done_indices]

    if done_indices:
        print(f"[*] Còn lại cần annotate: {len(todo):,} (đã skip {len(done_indices):,})")
    
    if not todo:
        print("[*] Tất cả đã xử lý! Gộp kết quả...")
    else:
        print(f"\n[*] Bắt đầu dán nhãn bằng DeepSeek / {MODEL}...")
        print(f"    (Concurrency={CONCURRENCY})")

        # Chạy annotate song song với semaphore
        async def process_one(idx: int, job_text: str) -> tuple[int, dict | None]:
            sample = await process_job(job_text, semaphore)
            save_progress_record(idx, sample)
            return idx, sample

        # Sử dụng tham số chạy một phần nếu được cấu hình (ví dụ: chạy thử 5 dòng đầu)
        # Để chạy thử 5 dòng, có thể set biến môi trường: RUN_TEST_ONLY=5
        test_limit = os.getenv('RUN_TEST_ONLY')
        if test_limit:
            test_limit = int(test_limit)
            todo = todo[:test_limit]
            print(f"[!] Chế độ TEST: Chỉ chạy {test_limit} dòng đầu.")

        tasks   = [process_one(idx, text) for idx, text in todo]
        results = await tqdm.gather(*tasks, desc="Dán nhãn văn bản")

        success = sum(1 for _, s in results if s is not None)
        print(f"\n[*] Đã dán nhãn thành công: {success:,} / {len(todo):,}")

    # Gộp tất cả sample từ progress file
    all_samples = load_progress_samples()

    pos = sum(1 for s in all_samples if len(s['label']) > 0)
    neg = len(all_samples) - pos

    # Thống kê label
    label_counts: Counter = Counter()
    for s in all_samples:
        for lbl_item in s['label']:
            label_counts[lbl_item[2]] += 1

    print(f"\n[*] Thống kê tập nhãn chuẩn (Gold Dataset):")
    print(f"    Positive samples (có entity): {pos:,}")
    print(f"    Negative samples (không có):  {neg:,}")
    print(f"    Tổng: {len(all_samples):,}")
    print(f"    Labels: {dict(label_counts)}")

    # Lưu output JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] XONG!")
    print(f"    Output Gold: {OUTPUT_FILE}")

if __name__ == '__main__':
    asyncio.run(main())
