import os
import sys
import json
import asyncio
import re
import warnings
from collections import Counter, defaultdict

from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx

sys.stdout.reconfigure(encoding='utf-8')

# ── OpenAI key ────────────────────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_base_dir, '.env'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("Error: Không tìm thấy OPENAI_API_KEY. Hãy copy .env.example → .env và điền key.")
    exit(1)

# ── Paths (override qua .env nếu cần) ────────────────────────────────────────
_project_root      = os.path.abspath(os.path.join(_base_dir, '..'))
_default_logs      = os.path.join(_project_root, 'standardization_job', 'logs')
_default_for_train = os.path.join(_base_dir, 'data', 'for_train.json')
_default_output    = os.path.join(_base_dir, 'data', 'train_dataset.json')

STD_LOGS_DIR   = os.getenv('STD_LOGS_DIR',   _default_logs)
FOR_TRAIN_FILE = os.getenv('FOR_TRAIN_FILE', _default_for_train)
OUTPUT_FILE    = os.getenv('OUTPUT_FILE',    _default_output)

# Resolve relative paths (tính từ thư mục chứa script)
for _var in ('STD_LOGS_DIR', 'FOR_TRAIN_FILE', 'OUTPUT_FILE'):
    _val = locals()[_var]
    if not os.path.isabs(_val):
        locals()[_var] = os.path.abspath(os.path.join(_base_dir, _val))
STD_LOGS_DIR   = STD_LOGS_DIR   if os.path.isabs(STD_LOGS_DIR)   else os.path.abspath(os.path.join(_base_dir, STD_LOGS_DIR))
FOR_TRAIN_FILE = FOR_TRAIN_FILE if os.path.isabs(FOR_TRAIN_FILE) else os.path.abspath(os.path.join(_base_dir, FOR_TRAIN_FILE))
OUTPUT_FILE    = OUTPUT_FILE    if os.path.isabs(OUTPUT_FILE)    else os.path.abspath(os.path.join(_base_dir, OUTPUT_FILE))

# ── Giới hạn theo provider (chỉ dùng trong fallback khi không có for_train.json) ──
PROVIDER_LIMITS: dict[str, int] = {
    'ITVIEC':      int(os.getenv('LIMIT_ITVIEC',   '1000')),
    'TOPCV':       int(os.getenv('LIMIT_TOPCV',    '3000')),
    'TOPDEV':      int(os.getenv('LIMIT_TOPDEV',   '2500')),
    'MBBANK':      int(os.getenv('LIMIT_MBBANK',   '1500')),
    '__default__': int(os.getenv('LIMIT_DEFAULT',  '500')),
}

# ── SSL Bypass (bỏ qua verify cert cho môi trường nội bộ / self-signed) ─────
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

_orig_httpx_init = httpx.AsyncClient.__init__
def _httpx_no_verify(self, *args, **kwargs):
    kwargs['verify'] = False
    _orig_httpx_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _httpx_no_verify

os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

CONCURRENCY_LIMIT = int(os.getenv('CONCURRENCY_LIMIT', '20'))
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

LABELS = ["MAJOR", "SKILL", "EXPERIENCE"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def tokenize_text(text: str) -> list[str]:
    """Tách từ và giữ dấu câu như những token riêng lẻ (tương thích GLiNER)."""
    return re.findall(r"[\w']+|[.,!?;()&]", text)


def find_token_indices(full_tokens: list[str], ent_tokens: list[str]) -> tuple[int, int]:
    """Tìm vị trí (start, end inclusive) của chuỗi token con trong chuỗi cha."""
    n = len(ent_tokens)
    for i in range(len(full_tokens) - n + 1):
        if full_tokens[i:i + n] == ent_tokens:
            return i, i + n - 1
    return -1, -1


# ── LLM annotation ───────────────────────────────────────────────────────────

async def extract_entities_from_text(text: str) -> list[dict]:
    """Gọi GPT-4o-mini với few-shot prompt publication-quality để trích xuất NER."""

    # ── System message ────────────────────────────────────────────────────────
    system_msg = (
        "You are an expert NER annotator for job descriptions (JD).\n"
        "Task: extract ALL entities of exactly 3 types: SKILL, MAJOR, EXPERIENCE.\n\n"
        "=== CRITICAL RULES ===\n"
        "1. VERBATIM: copy text EXACTLY as it appears — no paraphrasing, no abbreviation.\n"
        "2. EXHAUSTIVE: if a text lists 5 tools → annotate all 5; if 40 tools → annotate all 40.\n"
        "   Never stop early. Scan the ENTIRE text from start to finish.\n"
        "3. SEPARATE: each skill/tool/cert is its own entity.\n"
        "   'Java, Python, Go' → 3 entities, NOT 1.\n"
        "4. SKILL covers: programming languages, frameworks, databases, cloud services,\n"
        "   protocols, standards, methodologies, tools, soft skills, certifications,\n"
        "   foreign languages, and ANY domain-specific concepts (e.g., Marketing, Medical, Legal, HR, Logistics).\n"
        "5. MAJOR: academic field only (e.g., 'Computer Science', 'Kế toán', 'Luật').\n"
        "   NOT job titles, NOT company names.\n"
        "6. EXPERIENCE: work duration only (must contain a number, 'Không yêu cầu',\n"
        "   or 'Fresh graduate'). NOT project duration.\n"
        "7. Output JSON: {\"entities\": [{\"text\": \"...\", \"label\": \"SKILL|MAJOR|EXPERIENCE\"}]}"
    )

    # ── 5 Few-shot examples covering all major domains ────────────────────────
    ex1 = (
        "### EXAMPLE 1 — Embedded / Automotive\n"
        "TEXT: \"Develop AUTOSAR SWC/BSW in C/C++. Required: CAN, LIN, FlexRay, SOME/IP,"
        " DoIP, UDS, DCM, DEM, NvM, SecOC, ECU integration. Tools: Vector DaVinci, ETAS ISOLAR."
        " Standards: MISRA C, ASPICE, ISO 26262, ISO/SAE 21434. Dev tools: Git, Jenkins, GitLab,"
        " CI/CD, static analysis, unit testing, Agile/Scrum. 3+ years. B.S. Computer Science"
        " or Electrical Engineering.\"\n"
        '{"entities": ['
        '{"text": "AUTOSAR", "label": "SKILL"}, {"text": "SWC", "label": "SKILL"},'
        ' {"text": "BSW", "label": "SKILL"}, {"text": "C/C++", "label": "SKILL"},'
        ' {"text": "CAN", "label": "SKILL"}, {"text": "LIN", "label": "SKILL"},'
        ' {"text": "FlexRay", "label": "SKILL"}, {"text": "SOME/IP", "label": "SKILL"},'
        ' {"text": "DoIP", "label": "SKILL"}, {"text": "UDS", "label": "SKILL"},'
        ' {"text": "DCM", "label": "SKILL"}, {"text": "DEM", "label": "SKILL"},'
        ' {"text": "NvM", "label": "SKILL"}, {"text": "SecOC", "label": "SKILL"},'
        ' {"text": "ECU integration", "label": "SKILL"},'
        ' {"text": "Vector DaVinci", "label": "SKILL"}, {"text": "ETAS ISOLAR", "label": "SKILL"},'
        ' {"text": "MISRA C", "label": "SKILL"}, {"text": "ASPICE", "label": "SKILL"},'
        ' {"text": "ISO 26262", "label": "SKILL"}, {"text": "ISO/SAE 21434", "label": "SKILL"},'
        ' {"text": "Git", "label": "SKILL"}, {"text": "Jenkins", "label": "SKILL"},'
        ' {"text": "GitLab", "label": "SKILL"}, {"text": "CI/CD", "label": "SKILL"},'
        ' {"text": "static analysis", "label": "SKILL"}, {"text": "unit testing", "label": "SKILL"},'
        ' {"text": "Agile/Scrum", "label": "SKILL"},'
        ' {"text": "3+ years", "label": "EXPERIENCE"},'
        ' {"text": "Computer Science", "label": "MAJOR"},'
        ' {"text": "Electrical Engineering", "label": "MAJOR"}'
        "]}\n\n"
    )

    ex2 = (
        "### EXAMPLE 2 — Backend / Cloud (LONG TOOL LIST — annotate every single item)\n"
        "TEXT: \"Senior Backend Engineer, 5+ years. Tech stack: Golang, Python, gRPC, REST API,"
        " GraphQL, Microservices, Docker, Kubernetes, AWS, GCP, CircleCI, GitHub Actions,"
        " Terraform, Ansible, Amazon Aurora, MySQL, PostgreSQL, Elasticsearch, DynamoDB,"
        " Redis, Kafka, RabbitMQ, Datadog, Prometheus, Grafana, Swagger, Jira, Confluence,"
        " Git, SonarQube, OWASP, TDD, BDD, Agile. English. B.S. Computer Science.\"\n"
        '{"entities": ['
        '{"text": "Golang", "label": "SKILL"}, {"text": "Python", "label": "SKILL"},'
        ' {"text": "gRPC", "label": "SKILL"}, {"text": "REST API", "label": "SKILL"},'
        ' {"text": "GraphQL", "label": "SKILL"}, {"text": "Microservices", "label": "SKILL"},'
        ' {"text": "Docker", "label": "SKILL"}, {"text": "Kubernetes", "label": "SKILL"},'
        ' {"text": "AWS", "label": "SKILL"}, {"text": "GCP", "label": "SKILL"},'
        ' {"text": "CircleCI", "label": "SKILL"}, {"text": "GitHub Actions", "label": "SKILL"},'
        ' {"text": "Terraform", "label": "SKILL"}, {"text": "Ansible", "label": "SKILL"},'
        ' {"text": "Amazon Aurora", "label": "SKILL"}, {"text": "MySQL", "label": "SKILL"},'
        ' {"text": "PostgreSQL", "label": "SKILL"}, {"text": "Elasticsearch", "label": "SKILL"},'
        ' {"text": "DynamoDB", "label": "SKILL"}, {"text": "Redis", "label": "SKILL"},'
        ' {"text": "Kafka", "label": "SKILL"}, {"text": "RabbitMQ", "label": "SKILL"},'
        ' {"text": "Datadog", "label": "SKILL"}, {"text": "Prometheus", "label": "SKILL"},'
        ' {"text": "Grafana", "label": "SKILL"}, {"text": "Swagger", "label": "SKILL"},'
        ' {"text": "Jira", "label": "SKILL"}, {"text": "Confluence", "label": "SKILL"},'
        ' {"text": "Git", "label": "SKILL"}, {"text": "SonarQube", "label": "SKILL"},'
        ' {"text": "OWASP", "label": "SKILL"}, {"text": "TDD", "label": "SKILL"},'
        ' {"text": "BDD", "label": "SKILL"}, {"text": "Agile", "label": "SKILL"},'
        ' {"text": "English", "label": "SKILL"},'
        ' {"text": "5+ years", "label": "EXPERIENCE"},'
        ' {"text": "Computer Science", "label": "MAJOR"}'
        "]}\n\n"
    )

    ex3 = (
        "### EXAMPLE 3 — Data Science / AI / ML\n"
        "TEXT: \"Data Scientist, 3 years experience. Required: Python, R, SQL, TensorFlow,"
        " PyTorch, Scikit-learn, Keras, Pandas, NumPy, Matplotlib, Spark, Hadoop, Airflow,"
        " dbt, BigQuery, Redshift, Tableau, Power BI, MLflow, Kubeflow, Docker, Git,"
        " Statistics, Machine Learning, Deep Learning, NLP, Computer Vision, A/B testing."
        " Degree: Mathematics, Statistics, or Computer Science.\"\n"
        '{"entities": ['
        '{"text": "Python", "label": "SKILL"}, {"text": "R", "label": "SKILL"},'
        ' {"text": "SQL", "label": "SKILL"}, {"text": "TensorFlow", "label": "SKILL"},'
        ' {"text": "PyTorch", "label": "SKILL"}, {"text": "Scikit-learn", "label": "SKILL"},'
        ' {"text": "Keras", "label": "SKILL"}, {"text": "Pandas", "label": "SKILL"},'
        ' {"text": "NumPy", "label": "SKILL"}, {"text": "Matplotlib", "label": "SKILL"},'
        ' {"text": "Spark", "label": "SKILL"}, {"text": "Hadoop", "label": "SKILL"},'
        ' {"text": "Airflow", "label": "SKILL"}, {"text": "dbt", "label": "SKILL"},'
        ' {"text": "BigQuery", "label": "SKILL"}, {"text": "Redshift", "label": "SKILL"},'
        ' {"text": "Tableau", "label": "SKILL"}, {"text": "Power BI", "label": "SKILL"},'
        ' {"text": "MLflow", "label": "SKILL"}, {"text": "Kubeflow", "label": "SKILL"},'
        ' {"text": "Docker", "label": "SKILL"}, {"text": "Git", "label": "SKILL"},'
        ' {"text": "Statistics", "label": "SKILL"}, {"text": "Machine Learning", "label": "SKILL"},'
        ' {"text": "Deep Learning", "label": "SKILL"}, {"text": "NLP", "label": "SKILL"},'
        ' {"text": "Computer Vision", "label": "SKILL"}, {"text": "A/B testing", "label": "SKILL"},'
        ' {"text": "3 years experience", "label": "EXPERIENCE"},'
        ' {"text": "Mathematics", "label": "MAJOR"}, {"text": "Statistics", "label": "MAJOR"},'
        ' {"text": "Computer Science", "label": "MAJOR"}'
        "]}\n\n"
    )

    ex4 = (
        "### EXAMPLE 4 — Banking / Finance / Accounting (Vietnamese)\n"
        "TEXT: \"Yêu cầu: tốt nghiệp đại học chuyên ngành Tài chính - Ngân hàng, Kế toán,"
        " Kinh tế hoặc Quản trị kinh doanh. Tối thiểu 3 năm kinh nghiệm tại ngân hàng."
        " Thành thạo Excel, Word, PowerPoint, SQL, SAP, Oracle Financials, IFRS."
        " Chứng chỉ CPA, ACCA, CFA là lợi thế. Tiếng Anh thành thạo, TOEIC 700+."
        " Kỹ năng phân tích, giao tiếp, quản lý rủi ro, báo cáo tài chính.\"\n"
        '{"entities": ['
        '{"text": "Tài chính - Ngân hàng", "label": "MAJOR"},'
        ' {"text": "Kế toán", "label": "MAJOR"}, {"text": "Kinh tế", "label": "MAJOR"},'
        ' {"text": "Quản trị kinh doanh", "label": "MAJOR"},'
        ' {"text": "3 năm kinh nghiệm", "label": "EXPERIENCE"},'
        ' {"text": "Excel", "label": "SKILL"}, {"text": "Word", "label": "SKILL"},'
        ' {"text": "PowerPoint", "label": "SKILL"}, {"text": "SQL", "label": "SKILL"},'
        ' {"text": "SAP", "label": "SKILL"}, {"text": "Oracle Financials", "label": "SKILL"},'
        ' {"text": "IFRS", "label": "SKILL"}, {"text": "CPA", "label": "SKILL"},'
        ' {"text": "ACCA", "label": "SKILL"}, {"text": "CFA", "label": "SKILL"},'
        ' {"text": "Tiếng Anh", "label": "SKILL"}, {"text": "TOEIC 700+", "label": "SKILL"},'
        ' {"text": "phân tích", "label": "SKILL"}, {"text": "giao tiếp", "label": "SKILL"},'
        ' {"text": "quản lý rủi ro", "label": "SKILL"},'
        ' {"text": "báo cáo tài chính", "label": "SKILL"}'
        "]}\n\n"
    )

    ex5 = (
        "### EXAMPLE 5 — Front-end / Mobile / Design (Mixed EN-VI)\n"
        "TEXT: \"Tuyển Front-end Developer 2 năm kinh nghiệm. Yêu cầu: React, Vue.js, Angular,"
        " TypeScript, JavaScript, HTML5, CSS3, Sass/SCSS, Tailwind CSS, Webpack, Vite,"
        " React Native, Flutter, RESTful API, GraphQL, Jest, Cypress, Git, GitHub,"
        " Figma, Adobe XD, responsive design, performance optimization."
        " Tốt nghiệp Công nghệ thông tin hoặc Kỹ thuật phần mềm."
        " Tiếng Anh đọc hiểu tài liệu kỹ thuật. Teamwork, tư duy logic.\"\n"
        '{"entities": ['
        '{"text": "React", "label": "SKILL"}, {"text": "Vue.js", "label": "SKILL"},'
        ' {"text": "Angular", "label": "SKILL"}, {"text": "TypeScript", "label": "SKILL"},'
        ' {"text": "JavaScript", "label": "SKILL"}, {"text": "HTML5", "label": "SKILL"},'
        ' {"text": "CSS3", "label": "SKILL"}, {"text": "Sass/SCSS", "label": "SKILL"},'
        ' {"text": "Tailwind CSS", "label": "SKILL"}, {"text": "Webpack", "label": "SKILL"},'
        ' {"text": "Vite", "label": "SKILL"}, {"text": "React Native", "label": "SKILL"},'
        ' {"text": "Flutter", "label": "SKILL"}, {"text": "RESTful API", "label": "SKILL"},'
        ' {"text": "GraphQL", "label": "SKILL"}, {"text": "Jest", "label": "SKILL"},'
        ' {"text": "Cypress", "label": "SKILL"}, {"text": "Git", "label": "SKILL"},'
        ' {"text": "GitHub", "label": "SKILL"}, {"text": "Figma", "label": "SKILL"},'
        ' {"text": "Adobe XD", "label": "SKILL"},'
        ' {"text": "responsive design", "label": "SKILL"},'
        ' {"text": "performance optimization", "label": "SKILL"},'
        ' {"text": "Tiếng Anh", "label": "SKILL"}, {"text": "Teamwork", "label": "SKILL"},'
        ' {"text": "tư duy logic", "label": "SKILL"},'
        ' {"text": "2 năm kinh nghiệm", "label": "EXPERIENCE"},'
        ' {"text": "Công nghệ thông tin", "label": "MAJOR"},'
        ' {"text": "Kỹ thuật phần mềm", "label": "MAJOR"}'
        "]}\n\n"
    )

    few_shot = (
        ex1 + ex2 + ex3 + ex4 + ex5
    )

    ex6 = (
        "### EXAMPLE 6 — Sales / Marketing / Digital (Non-IT)\n"
        "TEXT: \"Tuyển Nhân viên Digital Marketing, 1-2 năm kinh nghiệm. Yêu cầu: tốt nghiệp Quản trị kinh doanh, Marketing hoặc Truyền thông. Có kinh nghiệm chạy Facebook Ads, Google Ads, TikTok Ads, SEO, SEM. Biết sử dụng Google Analytics, Canva, Photoshop cơ bản. Kỹ năng giao tiếp, lập kế hoạch, đàm phán tốt, chịu áp lực cao.\"\n"
        '{"entities": ['
        '{"text": "Digital Marketing", "label": "SKILL"},'
        ' {"text": "1-2 năm kinh nghiệm", "label": "EXPERIENCE"},'
        ' {"text": "Quản trị kinh doanh", "label": "MAJOR"}, {"text": "Marketing", "label": "MAJOR"},'
        ' {"text": "Truyền thông", "label": "MAJOR"},'
        ' {"text": "Facebook Ads", "label": "SKILL"}, {"text": "Google Ads", "label": "SKILL"},'
        ' {"text": "TikTok Ads", "label": "SKILL"}, {"text": "SEO", "label": "SKILL"},'
        ' {"text": "SEM", "label": "SKILL"}, {"text": "Google Analytics", "label": "SKILL"},'
        ' {"text": "Canva", "label": "SKILL"}, {"text": "Photoshop", "label": "SKILL"},'
        ' {"text": "giao tiếp", "label": "SKILL"}, {"text": "lập kế hoạch", "label": "SKILL"},'
        ' {"text": "đàm phán", "label": "SKILL"}, {"text": "chịu áp lực cao", "label": "SKILL"}'
        "]}\n\n"
    )

    ex7 = (
        "### EXAMPLE 7 — Medical / Healthcare\n"
        "TEXT: \"Tuyển Dược sĩ lâm sàng làm việc tại bệnh viện. Yêu cầu: bằng Bác sĩ hoặc Dược sĩ đại học trở lên. Kinh nghiệm 3 năm ở vị trí tương đương. Am hiểu Dược lý, quy định Bộ Y Tế, GPP, GMP. Kỹ năng tư vấn thuốc, giao tiếp bệnh nhân, xử lý tình huống y khoa khẩn cấp. Tiếng Anh chuyên ngành tốt.\"\n"
        '{"entities": ['
        '{"text": "Dược sĩ lâm sàng", "label": "SKILL"},'
        ' {"text": "Bác sĩ", "label": "MAJOR"}, {"text": "Dược sĩ đại học", "label": "MAJOR"},'
        ' {"text": "Kinh nghiệm 3 năm", "label": "EXPERIENCE"},'
        ' {"text": "Dược lý", "label": "SKILL"}, {"text": "quy định Bộ Y Tế", "label": "SKILL"},'
        ' {"text": "GPP", "label": "SKILL"}, {"text": "GMP", "label": "SKILL"},'
        ' {"text": "tư vấn thuốc", "label": "SKILL"}, {"text": "giao tiếp bệnh nhân", "label": "SKILL"},'
        ' {"text": "xử lý tình huống y khoa", "label": "SKILL"},'
        ' {"text": "Tiếng Anh chuyên ngành", "label": "SKILL"}'
        "]}\n\n"
    )

    ex8 = (
        "### EXAMPLE 8 — HR / Admin / Legal / Logistics\n"
        "TEXT: \"Chuyên viên C&B (Nhân sự) 4+ năm, hoặc nhân sự Logistics. Yêu cầu: Cử nhân Quản trị nguồn nhân lực, Luật hoặc Logistics. Nắm vững Luật Lao động, BHXH, Thuế TNCN hoặc Incoterms, Hải quan. Sử dụng thành thạo phần mềm AMIS, SAP HR. Kỹ năng tin học văn phòng xuất sắc. Có kỹ năng giải quyết xung đột, tư duy phân tích.\"\n"
        '{"entities": ['
        '{"text": "C&B", "label": "SKILL"}, {"text": "Nhân sự", "label": "SKILL"},'
        ' {"text": "Logistics", "label": "SKILL"},'
        ' {"text": "4+ năm", "label": "EXPERIENCE"},'
        ' {"text": "Quản trị nguồn nhân lực", "label": "MAJOR"}, {"text": "Luật", "label": "MAJOR"},'
        ' {"text": "Logistics", "label": "MAJOR"},'
        ' {"text": "Luật Lao động", "label": "SKILL"}, {"text": "BHXH", "label": "SKILL"},'
        ' {"text": "Thuế TNCN", "label": "SKILL"}, {"text": "Incoterms", "label": "SKILL"},'
        ' {"text": "Hải quan", "label": "SKILL"}, {"text": "AMIS", "label": "SKILL"},'
        ' {"text": "SAP HR", "label": "SKILL"}, {"text": "tin học văn phòng", "label": "SKILL"},'
        ' {"text": "giải quyết xung đột", "label": "SKILL"}, {"text": "tư duy phân tích", "label": "SKILL"}'
        "]}\n\n"
    )

    few_shot = (
        ex1 + ex2 + ex3 + ex4 + ex5 + ex6 + ex7 + ex8
        + "### NOW ANNOTATE — scan the ENTIRE text, do NOT stop until all entities found\n"
        + f'TEXT: "{text}"\n'
        + "OUTPUT:"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": few_shot},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4096,
        )
        data = json.loads(response.choices[0].message.content)
        if "entities" in data and isinstance(data["entities"], list):
            return data["entities"]
        for val in data.values():
            if isinstance(val, list):
                return val
        return []
    except Exception:
        return []



async def process_job(job_text: str, sem: asyncio.Semaphore) -> dict | None:
    """Annotate một đoạn text và trả về {"tokenized_text": [...], "ner": [...]}."""
    async with sem:
        if not job_text or len(job_text.strip()) < 10:
            return None

        llm_entities = await extract_entities_from_text(job_text)
        if not llm_entities:
            return None

        tokenized_text = tokenize_text(job_text)
        ner: list[list] = []

        for ent in llm_entities:
            ent_text = ent.get('text', '')
            label    = ent.get('label', '').upper()
            if label not in LABELS or not ent_text:
                continue

            ent_tokens = tokenize_text(ent_text)
            if not ent_tokens:
                continue

            start_idx, end_idx = find_token_indices(tokenized_text, ent_tokens)
            if start_idx != -1:
                ner.append([start_idx, end_idx, label])

        if ner:
            return {"tokenized_text": tokenized_text, "ner": ner}
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_text_from_job(job: dict) -> str:
    """Gộp tags, majors, description, requirements, experience thành 1 chuỗi có nhãn."""
    bi = job.get('basic_info') or {}
    dc = job.get('display_content') or {}

    tags     = bi.get('tags') or []
    majors   = bi.get('majors') or []
    raw_desc = (dc.get('raw_description')    or '').strip()
    raw_req  = (dc.get('raw_requirements')   or '').strip()
    raw_exp  = (dc.get('raw_experience_text') or '').strip()

    parts: list[str] = []
    if tags:     parts.append(f"[TAGS]: {', '.join(str(t) for t in tags)}")
    if majors:   parts.append(f"[MAJORS]: {', '.join(str(m) for m in majors)}")
    if raw_desc: parts.append(f"[DESCRIPTION]:\n{raw_desc}")
    if raw_req:  parts.append(f"[REQUIREMENTS]:\n{raw_req}")
    if raw_exp:  parts.append(f"[EXPERIENCE]:\n{raw_exp}")

    return "\n\n".join(parts).strip()


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    texts_to_process: list[str] = []

    # ── Ưu tiên đọc for_train.json (pre-sampled bởi sample_for_train.py) ──────
    if os.path.isfile(FOR_TRAIN_FILE):
        print(f"[*] Đọc pre-sampled data từ: {FOR_TRAIN_FILE}")
        with open(FOR_TRAIN_FILE, 'r', encoding='utf-8') as f:
            records: list[dict] = json.load(f)
        print(f"    Tổng records: {len(records):,}")

        provider_counts: Counter = Counter()
        for rec in records:
            text = (rec.get('text') or '').strip()
            if len(text) > 30:
                texts_to_process.append(text)
                provider_counts[rec.get('provider', 'UNKNOWN')] += 1

        for prov, cnt in sorted(provider_counts.items()):
            print(f"  [{prov}] {cnt:,} texts sẽ được annotate")

    else:
        # ── Fallback: quét toàn bộ STD_LOGS_DIR ──────────────────────────────
        print(f"[!] Không tìm thấy {FOR_TRAIN_FILE}")
        print(f"    → Hãy chạy: python ../standardization_job/sample_for_train.py")
        print(f"[*] Fallback — Loading from: {STD_LOGS_DIR}")

        all_jobs: list[dict] = []
        for fname in os.listdir(STD_LOGS_DIR):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(STD_LOGS_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                jobs = data if isinstance(data, list) else []
                all_jobs.extend(jobs)
                print(f"  Loaded {len(jobs):,} jobs from {fname}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Skipped {fname}: {e}")

        if not all_jobs:
            print("[!] Không có job nào để xử lý.")
            return

        print(f"[*] Total jobs: {len(all_jobs):,}")

        buckets: dict[str, list] = defaultdict(list)
        for job in all_jobs:
            provider = (job.get('source_metadata') or {}).get('provider', 'UNKNOWN').upper()
            buckets[provider].append(job)

        for provider, jobs in buckets.items():
            limit = PROVIDER_LIMITS.get(provider, PROVIDER_LIMITS['__default__'])
            taken = jobs[:limit]
            print(f"  [{provider}] {len(jobs):,} → lấy {len(taken):,} (limit={limit:,})")
            for job in taken:
                text = _build_text_from_job(job)
                if len(text) > 30:
                    texts_to_process.append(text)

    print(f"[*] Valid texts to annotate: {len(texts_to_process):,}")

    if not texts_to_process:
        print("[!] Không có text nào để annotate.")
        return

    # ── Annotate bằng GPT-4o-mini ─────────────────────────────────────────────
    print(f"[*] Annotating với GPT-4o-mini (concurrency={CONCURRENCY_LIMIT})...")
    tasks   = [process_job(t, semaphore) for t in texts_to_process]
    results = await tqdm.gather(*tasks)

    dataset = [r for r in results if r is not None]
    print(f"[*] Annotated thành công: {len(dataset):,} samples")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] THÀNH CÔNG!")
    print(f"    - Tổng samples : {len(dataset):,}")
    print(f"    - Đã lưu vào   : {OUTPUT_FILE}")


if __name__ == '__main__':
    asyncio.run(main())
