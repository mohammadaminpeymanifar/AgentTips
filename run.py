# =========================================================
# بخش ۱: وارد کردن کتابخانه‌ها و ماژول‌های مورد نیاز
# =========================================================

import os  # ماژول دسترسی به سیستم‌عامل برای ساخت پوشه‌ها و خواندن مسیرها
import json  # ماژول کار با داده‌ها و فایل‌های فرمت JSON
import logging  # ماژول ثبت لاگ‌ها و پیام‌های وضعیت برنامه در ترمینال
import urllib.request  # ماژول ارسال درخواست‌های HTTP/HTTPS برای دانلود تصاویر و ارسال به تلگرام
import urllib.parse  # ماژول کدگذاری پارامترهای آدرس‌های وب (URL Encoding)
from datetime import datetime  # ماژول دریافت تاریخ و زمان فعلی سیستم
from typing import List, Literal, Optional  # ماژول تعریف تایپ‌های مشخص برای متغیرها و توابع در پایتون
from dotenv import load_dotenv  # ماژول بارگذاری متغیرهای محیطی از فایل .env
from pydantic import BaseModel, Field  # ماژول ساخت ساختار داده‌ای معتبر و صحه‌گذاری شده (Schema)
from tenacity import retry, stop_after_attempt, wait_exponential  # ماژول مدیریت تلاش مجدد خودکار در صورت بروز خطا

import chromadb  # دیتابیس برداری لوکال برای ذخیره و جستجوی متن‌ها بر اساس شباهت معنایی
from chromadb.utils import embedding_functions  # توابع تبدیل متن به بردار عددی (Embedding) در ChromaDB

from openai import OpenAI  # کلاینت رسمی برای ارتباط با APIهای سازگار با OpenAI (مثل AvalAI)
from langchain_openai import ChatOpenAI  # کلاینت مدل‌های زبانی Chat از فریم‌ورک LangChain
from langchain_community.tools import DuckDuckGoSearchRun  # ابزار جستجوی موتور DuckDuckGo در LangChain
from langchain_core.messages import SystemMessage, HumanMessage  # کلاس‌های ساخت پیام سیستم و پیام کاربر برای LLM

# =========================================================
# بخش ۲: پیکربندی لاگ‌گیری و متغیرهای محیطی
# =========================================================

# تنظیم نحوه نمایش لاگ‌ها در ترمینال (زمان، سطح لاگ و متن پیام)
logging.basicConfig(
    level=logging.INFO,  # نمایش پیام‌های سطح INFO و بالاتر ( مثل WARNING و ERROR)
    format="%(asctime)s [%(levelname)s] %(message)s",  # فرمت نمایش: زمان [سطح لاگ] متن
    datefmt="%Y-%m-%d %H:%M:%S"  # فرمت تاریخ و زمان
)
logger = logging.getLogger("TipsAgent")  # ایجاد یک لاگر اختصاصی با نام TipsAgent

# خواندن متغیرهای موجود در فایل .env و بارگذاری آن‌ها در سیستم‌عامل
load_dotenv()

# استخراج کلیدهای API و تنظیمات از متغیرهای محیطی
AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")  # کلید دسترسی به سرویس AvalAI
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # توکن ربات تلگرام
CHANNEL = os.getenv("TELEGRAM_CHANNEL")  # آیدی یا یوزرنیم کانال تلگرام Target

# بررسی وجود کلیدها و توقف برنامه در صورت عدم وجود آن‌ها
if not AVALAI_API_KEY:
    raise ValueError("کلید AVALAI_API_KEY در فایل .env یافت نشد.")  # پرتاب خطا در صورت نبود کلید اول
if not BOT_TOKEN or not CHANNEL:
    raise ValueError("متغیرهای TELEGRAM_BOT_TOKEN یا TELEGRAM_CHANNEL در فایل .env یافت نشدند.")  # پرتاب خطا در صورت نبود اطلاعات تلگرام

# تعریف مسیرهای ذخیره‌سازی فایل‌ها و دیتابیس
DATA_DIR = "./data"  # پوشه اصلی داده‌ها
IMAGES_DIR = os.path.join(DATA_DIR, "images")  # پوشه ذخیره تصاویر دانلود شده
JSON_PATH = os.path.join(DATA_DIR, "posted_tips.json")  # مسیر فایل تاریخچه JSON
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")  # مسیر ذخیره‌سازی فایل‌های دیتابیس ChromaDB

# ساخت خودکار پوشه‌های مورد نیاز در صورت عدم وجود
os.makedirs(DATA_DIR, exist_ok=True)  # ساخت پوشه data (اگر وجود ندارد)
os.makedirs(IMAGES_DIR, exist_ok=True)  # ساخت پوشه images (اگر وجود ندارد)

# =========================================================
# بخش ۳: تعریف ساختار داده‌ها (Pydantic Schemas)
# =========================================================

# ساختار یک نکته هوش مصنوعی (Agent Tip)
class AgentTip(BaseModel):
    title_en: str = Field(description="Short title in English.")  # عنوان انگلیسی نکته
    body_en: str = Field(description="Detailed explanation in English.")  # توضیحات کامل انگلیسی
    title_fa: str = Field(description="Natural, fluent Persian title like an expert tech educator.")  # عنوان فارسی روان
    body_fa: str = Field(description="Natural, fluent Persian body explanation avoiding direct word-for-word translation.")  # متن فارسی روان
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] = Field(description="Difficulty level.")  # سطح سختی نکته
    source_url: Optional[str] = Field(default=None, description="Source URL or document link.")  # لینک منبع (اختیاری)

# ساختار خروجی چندتایی برای تولید کاندیداها
class TipCandidates(BaseModel):
    tips: List[AgentTip] = Field(
        description="A list of 3 to 5 distinct agentic AI tip candidates.",  # توضیحات فیلد
        min_items=3,  # حداقل ۳ کاندیدا
        max_items=5   # حداکثر ۵ کاندیدا
    )

# =========================================================
# بخش ۴: راه‌اندازی دیتابیس برداری و کلاینت‌ها
# =========================================================

# ساخت تابع تبدیل متن به بردار (Default Embedding)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# ایجاد یا اتصال به دیتابیس برداری دائمی ChromaDB در مسیر مشخص شده
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ایجاد یا دریافت کالکشن posted_tips برای ذخیره بردارها با معیار شباهت کسینوسی
collection = chroma_client.get_or_create_collection(
    name="posted_tips",  # نام مجموعه
    embedding_function=embedding_fn,  # تابع امبدینگ
    metadata={"hnsw:space": "cosine"}  # محاسبه شباهت بر اساس فاصله کسینوسی
)

# تعریف کلاینت OpenAI جهت اتصال به سرویس AvalAI برای بخش‌های غیر LangChain (مثل تصویر)
openai_client = OpenAI(
    api_key=AVALAI_API_KEY,  # کلید API
    base_url="https://api.avalai.ir/v1"  # آدرس سرور AvalAI
)

# =========================================================
# بخش ۵: توابع فراخوانی سرویس‌های خارجی همراه با Retry
# =========================================================

# تابع اجرای جستجو در وب با قابلیت ۳ بار تلاش مجدد در صورت قطع ارتباط
@retry(
    stop=stop_after_attempt(3),  # حداکثر ۳ بار تلاش در صورت خطا
    wait=wait_exponential(multiplier=1, min=2, max=10),  # زمان انتظار بین تلاش‌ها (۲ تا ۱۰ ثانیه)
    reraise=True  # ارسال مجدد خطا در صورت شکست پس از ۳ بار تلاش
)
def fetch_search_results(search_tool: DuckDuckGoSearchRun, query: str) -> str:
    logger.info(f"[SEARCH] Searching web via DuckDuckGo: '{query}'")  # لاگ‌گیری جستجو
    return search_tool.invoke(query)  # اجرای وب‌سرچ و بازگرداندن متن نتایج

# تابع درخواست تولید نکته‌ها از هوش مصنوعی با قابلیت Retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_candidates_with_llm(structured_llm, messages) -> TipCandidates:
    logger.info("[LLM] Generating candidate tips via AvalAI (DeepSeek)...")  # لاگ شروع کار LLM
    return structured_llm.invoke(messages)  # ارسال پیام‌ها به مدل و دریافت خروجی ساختاریافته

# تابع درخواست ساخت تصویر از DALL-E 3 با قابلیت Retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_image_api(prompt: str):
    logger.info("[IMAGE] Requesting visual generation from AvalAI Image API...")  # لاگ ساخت تصویر
    return openai_client.images.generate(  # فراخوانی API تولید تصویر
        model="dall-e-3",  # استفاده از مدل DALL-E 3
        prompt=prompt,  # توصیف تصویر
        size="1024x1024",  # ابعاد تصویر
        quality="standard",  # کیفیت استاندارد
        n=1,  # تعداد تصویر (۱ عدد)
    )

# تابع ارسال تصویر و متن به تلگرام با قابلیت Retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def send_telegram_photo(url: str, request_data: bytes, boundary: str):
    logger.info("[TELEGRAM] Dispatching photo & caption to Telegram channel...")  # لاگ ارسال به تلگرام
    req = urllib.request.Request(  # ساخت شیء درخواست HTTP
        url,  # آدرس ای‌پس‌آی تلگرام
        data=request_data,  # بایت‌های داده‌های ارسال
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}  # هدر مشخص‌کننده فرم مالتی‌پارت
    )
    with urllib.request.urlopen(req) as response:  # ارسال درخواست و باز کردن پاسخ
        return response.read().decode("utf-8")  # خواندن پاسخ تلگرام به صورت متنی

# =========================================================
# بخش ۶: توابع منطق برنامه، حافظه و بررسی تکرار
# =========================================================

# تابع خواندن سطح سختی ۲ پست اخیر از روی فایل JSON
def get_last_two_difficulties() -> List[str]:
    if not os.path.exists(JSON_PATH):  # اگر فایل هنوز ساخته نشده بود
        return []  # لیست خالی برگردان
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:  # باز کردن فایل JSON
            data = json.load(f)  # خواندن محتوای JSON
            if not data:  # اگر فایل خالی بود
                return []
            return [t.get("difficulty") for t in data[-2:] if "difficulty" in t]  # استخراج سختی ۲ عنصر آخر
    except Exception as e:  # در صورت وجود هرگونه خطای خواندن
        logger.warning(f"Could not read posted_tips.json: {e}")  # لاگ هشدار
        return []  # بازگرداندن لیست خالی

# تابع بررسی شباهت متنی با پست‌های قبلی جهت جلوگیری از تکرار
def is_duplicate(tip: AgentTip, threshold: float = 0.85) -> tuple[bool, float]:
    if collection.count() == 0:  # اگر دیتابیس برداری هنوز خالی است
        return False, 0.0  # تکراری نیست

    text_to_check = f"{tip.title_en} {tip.body_en}"  # ترکیب عنوان و متن انگلیسی برای بررسی
    results = collection.query(query_texts=[text_to_check], n_results=1)  # جستجوی شبیه‌ترین متن در دیتابیس

    if results and results["distances"] and results["distances"][0]:  # اگر خلاصه‌ای یافت شد
        distance = results["distances"][0][0]  # استخراج فاصله برداری
        similarity = 1.0 - distance  # تبدیل فاصله به درصد شباهت (از ۰ تا ۱)
        return similarity >= threshold, similarity  # اگر شباهت >= ۰.۸۵ بود یعنی تکراری است

    return False, 0.0  # در غیر این صورت تکراری نیست

# تابع انتخاب بهترین کاندیدا بر اساس تنوع سطح سختی
def select_best_candidate(candidates: List[AgentTip]) -> Optional[AgentTip]:
    if not candidates:  # اگر لیست کاندیداهای زنده خالی بود
        return None

    recent_difficulties = get_last_two_difficulties()  # خواندن سختی پست‌های اخیر
    logger.info(f"[DIVERSITY] Recent difficulties in memory: {recent_difficulties}")  # لاگ سطح سختی‌های اخیر

    # جداسازی گزینه‌هایی که سطح سختی متفاوتی با ۲ پست اخیر دارند
    diverse = [c for c in candidates if c.difficulty not in recent_difficulties]
    if diverse:  # اگر گزینه با سختی جدید یافت شد
        logger.info(f"[SELECT] Picked candidate with DIVERSE difficulty ({diverse[0].difficulty}): '{diverse[0].title_en}'")  # لاگ انتخاب
        return diverse[0]  # بازگرداندن اولین گزینه متنوع

    # اگر همه هم‌سطح بودند، اولین کاندیدا به عنوان پشتیبان انتخاب می‌شود
    logger.info(f"[SELECT] Fallback to top-relevant candidate ({candidates[0].difficulty}): '{candidates[0].title_en}'")
    return candidates[0]

# تابع ساخت پرامپت پویا و دانلود تصویر DALL-E 3
def create_and_download_image(tip: AgentTip) -> str:
    # ساخت توصیف متنی تصویر بر اساس عنوان و متن نکته
    image_prompt = (
        f"Futuristic sci-fi editorial illustration representing the tech concept: '{tip.title_en}'. "
        f"Specific focal metaphor: {tip.body_en}. "
        "Clean 2.5D isometric-leaning style, soft volumetric lighting, deep layered planes, "
        "futuristic translucent materials, vibrant glowing accents. "
        "Educational visual metaphor, single clear focal subject. "
        "NO mascot robot, NO human face, NO text, NO letters, NO words, high resolution, 3D render feel."
    )

    response = generate_image_api(image_prompt)  # درخواست تصویر از API
    image_url = response.data[0].url  # استخراج لینک دانلود تصویر

    timestamp = int(datetime.now().timestamp())  # ساخت برچسب زمانی فرضی برای نام‌گذاری
    image_path = os.path.join(IMAGES_DIR, f"tip_{timestamp}.png")  # ساخت مسیر نهایی ذخیره عکس

    logger.info(f"[IMAGE] Downloading generated visual to {image_path}...")  # لاگ شروع دانلود
    urllib.request.urlretrieve(image_url, image_path)  # دانلود فایل عکس و ذخیره در مسیر
    return image_path  # بازگرداندن آدرس محلی عکس دانلود شده

# تابع قالب‌بندی و ارسال پیام به همراه تصویر به تلگرام
def post_to_telegram(tip: AgentTip, image_path: str):
    # فرمت‌دهی متن کپشن پست به شکل دو زبانه و HTML
    caption = (
        f"<b>{tip.title_en}</b>\n"  # عنوان انگلیسی پررنگ
        f"{tip.body_en}\n\n"  # متن انگلیسی
        f"<b>{tip.title_fa}</b>\n"  # عنوان فارسی پررنگ
        f"{tip.body_fa}\n\n"  # متن فارسی
        f"See you tomorrow for another AI agents tip!\n"  # متن پایانی
        f"{CHANNEL}"  # لینک/یوزرنیم کانال
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"  # آدرس اکشن sendPhoto ربات تلگرام
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"  # مرزبندی بخش‌های درخواست Multipart

    with open(image_path, "rb") as f:  # خواندن بایت‌های فایل تصویر
        image_bytes = f.read()

    # ساخت بدنه درخواست به فرمت multipart/form-data
    body = bytearray()
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{CHANNEL}\r\n".encode("utf-8"))
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{os.path.basename(image_path)}\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8"))
    body.extend(image_bytes)  # قرار دادن فایل عکس
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))  # انتهای فرم

    send_telegram_photo(url, bytes(body), boundary)  # ارسال نهایی درخواست به تلگرام
    logger.info("[TELEGRAM] Message and visual posted successfully!")  # لاگ ارسال موفق

# تابع ذخیره پست ارسال‌شده در دیتابیس ChromaDB و فایل JSON
def save_posted_tip(tip: AgentTip, image_path: str):
    timestamp = datetime.now().isoformat()  # ثبت زمان فعلی به فرمت ISO
    doc_text = f"{tip.title_en} {tip.body_en}"  # متن ترکیبی برای امبدینگ
    doc_id = f"tip_{int(datetime.now().timestamp())}"  # شناسه یکتا برای سند در دیتابیس

    # اضافه کردن سند جدید به دیتابیس برداری ChromaDB
    collection.add(
        documents=[doc_text],  # متن
        metadatas=[{"title_en": tip.title_en, "date": timestamp}],  # متاداده‌ها
        ids=[doc_id]  # آیدی
    )

    json_data = []  # ایجاد لیست داده‌های JSON
    if os.path.exists(JSON_PATH):  # اگر فایل سابقه وجود دارد
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                json_data = json.load(f)  # خواندن سوابق قبلی
        except Exception:
            json_data = []  # در صورت وجود خطا لیست خالی می‌شود

    # تبدیل شیء Pydantic به دیکشنری پایتون و اضافه کردن تاریخ و آدرس تصویر
    tip_entry = tip.model_dump()
    tip_entry["posted_date"] = timestamp
    tip_entry["image_path"] = image_path
    json_data.append(tip_entry)  # اضافه کردن پست جدید به انتهای لیست

    # بازنویسی و بروزرسانی فایل posted_tips.json
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info("[MEMORY] Tip details stored in ChromaDB and JSON log.")  # لاگ ذخیره حافظه

# =========================================================
# بخش ۷: نقطه ورود و اجرای زنجیره ایجنت (Pipeline)
# =========================================================

def run_pipeline():
    logger.info("=== STARTING TIPS AGENT PIPELINE ===")  # لاگ شروع کلی خط لوله

    try:
        search_tool = DuckDuckGoSearchRun()  # ساخت ابزار جستجو
        llm = ChatOpenAI(  # تنظیم مدل زبانی
            model="deepseek-chat",  # مدل DeepSeek در AvalAI
            api_key=AVALAI_API_KEY,  # کلید API
            base_url="https://api.avalai.ir/v1",  # آدرس سرور
            temperature=0.7  # میزان خلاقیت
        )

        # گام ۱: اجرای وب‌سرچ برای دریافت آخرین ایده‌ها
        query = "latest agentic AI architecture design patterns 2026"  # عبارت جستجو
        search_results = fetch_search_results(search_tool, query)  # انجام جستجو

        # گام ۲: تولید ۳ تا ۵ کاندیدا توسط مدل زبانی
        system_prompt = (
            "You are an expert AI Agents Educator. Ground your insights on search results. "
            "Ensure title_fa and body_fa sound extremely natural, idiomatic, and educator-like."
        )
        messages = [
            SystemMessage(content=system_prompt),  # دستورالعمل سیستم
            HumanMessage(content=f"Search findings:\n{search_results}\n\nGenerate 3 to 5 candidate tips.")  # متغیر ورود نتایج
        ]
        structured_llm = llm.with_structured_output(TipCandidates)  # تحمیل فرمت خروجی Pydantic
        result: TipCandidates = generate_candidates_with_llm(structured_llm, messages)  # دریافت خروجی
        logger.info(f"[GENERATION] Generated {len(result.tips)} raw candidates.")  # لاگ تعداد تولیدشده‌ها

        # گام ۳: فیلتر کاندیداهای تکراری بر اساس شباهت معنایی
        surviving_candidates = []  # لیست کاندیداهای قبول‌شده
        for idx, tip in enumerate(result.tips, 1):
            duplicate, score = is_duplicate(tip, threshold=0.85)  # تست تکراری بودن
            if duplicate:
                logger.info(f"[DEDUP] Candidate #{idx} DROPPED (Similarity: {score:.2f} >= 0.85): '{tip.title_en}'")  # لاگ رد شدن
            else:
                logger.info(f"[DEDUP] Candidate #{idx} PASSED (Similarity: {score:.2f} < 0.85): '{tip.title_en}'")  # لاگ قبول شدن
                surviving_candidates.append(tip)  # افزودن به لیست قبول‌شده‌ها

        # گام ۴: انتخاب بهترین کاندیدا بر اساس تنوع سطح سختی
        best_tip = select_best_candidate(surviving_candidates)

        if not best_tip:  # اگر هیچ گزینه‌ای باقی نمانده بود (همه تکراری بودند)
            logger.warning("[PIPELINE] No unique candidates survived deduplication. Skipping post today.")  # لاگ لغو برنامه
            return  # توقف خط لوله

        # گام ۵: تولید تصویر متناسب با نکته برنده
        image_path = create_and_download_image(best_tip)

        # گام ۶: ارسال متن دو زبانه و تصویر به کانال تلگرام
        post_to_telegram(best_tip, image_path)

        # گام ۷: ثبت و ذخیره پست ارسال‌شده در دیتابیس برداری و JSON
        save_posted_tip(best_tip, image_path)

        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")  # لاگ اتمام موفقیت‌آمیز

    except Exception as e:  # مدیریت خطاهای غیرمنتظره در کل پیپ‌لاین
        logger.error(f"[PIPELINE FAILED] Critical error: {e}", exc_info=True)  # ثبت خطا با جزئیات کامل traceback

# نقطه ورود اجرایی پایتون
if __name__ == "__main__":
    run_pipeline()  # اجرای تابع اصلی