import os
import json
import urllib.request
from datetime import datetime
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import chromadb
from chromadb.utils import embedding_functions

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# بارگذاری متغیرهای محیطی
load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
if not AVALAI_API_KEY:
    raise ValueError("کلید AVALAI_API_KEY در فایل .env پیدا نشد.")

# مسیرهای داده و تصاویر
DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSON_PATH = os.path.join(DATA_DIR, "posted_tips.json")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ۱. مدل‌های Pydantic
class AgentTip(BaseModel):
    title_en: str = Field(description="Short title in English.")
    body_en: str = Field(description="Detailed explanation in English.")
    title_fa: str = Field(description="Natural, fluent Persian title.")
    body_fa: str = Field(description="Natural, fluent Persian body explanation.")
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] = Field(description="Difficulty level.")
    source_url: Optional[str] = Field(default=None, description="Source URL or document link.")

class TipCandidates(BaseModel):
    tips: List[AgentTip] = Field(
        description="A list of 3 to 5 distinct agentic AI tip candidates.",
        min_items=3,
        max_items=5
    )

# ۲. راه‌اندازی ChromaDB
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="posted_tips",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# کلاینت مستقیم OpenAI برای فراخوانی API تصویر AvalAI
openai_client = OpenAI(
    api_key=AVALAI_API_KEY,
    base_url="https://api.avalai.ir/v1"
)

def get_last_two_difficulties() -> List[str]:
    """خواندن سطح سختی دو پست اخیر از فایل JSON"""
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data:
                return []
            recent_tips = data[-2:]
            return [t.get("difficulty") for t in recent_tips if "difficulty" in t]
    except Exception:
        return []

def is_duplicate(tip: AgentTip, threshold: float = 0.85) -> tuple[bool, float]:
    """بررسی شباهت کسینوسی در ChromaDB"""
    if collection.count() == 0:
        return False, 0.0

    text_to_check = f"{tip.title_en} {tip.body_en}"
    results = collection.query(
        query_texts=[text_to_check],
        n_results=1
    )

    if results and results["distances"] and results["distances"][0]:
        distance = results["distances"][0][0]
        similarity = 1.0 - distance
        if similarity >= threshold:
            return True, similarity
        return False, similarity

    return False, 0.0

def select_best_candidate(candidates: List[AgentTip]) -> Optional[AgentTip]:
    """انتخاب بهترین کاندیدا بر اساس اولویت تنوع سطح سختی و سپس تناسب"""
    if not candidates:
        return None

    recent_difficulties = get_last_two_difficulties()
    print(f"\n📊 Recent 2 posted difficulties: {recent_difficulties}")

    diverse_candidates = [
        c for c in candidates if c.difficulty not in recent_difficulties
    ]

    if diverse_candidates:
        selected = diverse_candidates[0]
        print(f"🎯 Selected candidate with DIVERSE difficulty ({selected.difficulty}): '{selected.title_en}'")
        return selected

    selected = candidates[0]
    print(f"⚠️ Fallback to top-relevant candidate ({selected.difficulty}): '{selected.title_en}'")
    return selected

def generate_and_save_image(tip: AgentTip) -> str:
    """ساخت پرامپت اختصاصی و تولید تصویر با استایل 2.5D Isometric Sci-Fi و ذخیره آن"""
    print("\n--- Generating Visual Metaphor Image via AvalAI ---")
    
    # ساخت پرامپت تصویر بر اساس محتوای اختصاصی نکته
    image_prompt = (
        f"Futuristic sci-fi editorial illustration representing the tech concept: '{tip.title_en}'. "
        f"Specific focal metaphor: {tip.body_en}. "
        "Clean 2.5D isometric-leaning style, soft volumetric lighting, deep layered planes, "
        "futuristic translucent materials, vibrant glowing accents. "
        "Educational visual metaphor, single clear focal subject. "
        "NO mascot robot, NO human face, NO text, NO letters, NO words, high resolution, 3D render feel."
    )

    print(f"🎨 Image Prompt: {image_prompt}\n")

    # فراخوانی endpoint تولید تصویر
    response = openai_client.images.generate(
        model="dall-e-3",  # یا مدل تصویرساز فعال در AvalAI
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    timestamp = int(datetime.now().timestamp())
    image_filename = f"tip_{timestamp}.png"
    image_path = os.path.join(IMAGES_DIR, image_filename)

    # دانلود و ذخیره تصویر در مسیر ./data/images/
    print(f"⬇️ Downloading image to {image_path}...")
    urllib.request.urlretrieve(image_url, image_path)
    print("✅ Image successfully generated and saved!")

    return image_path

def save_posted_tip(tip: AgentTip, image_path: str):
    """ذخیره اطلاعات نکته و مسیر تصویر در ChromaDB و JSON"""
    timestamp = datetime.now().isoformat()
    doc_text = f"{tip.title_en} {tip.body_en}"
    doc_id = f"tip_{int(datetime.now().timestamp())}"

    collection.add(
        documents=[doc_text],
        metadatas=[{"title_en": tip.title_en, "date": timestamp}],
        ids=[doc_id]
    )

    json_data = []
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception:
            json_data = []

    tip_entry = tip.model_dump()
    tip_entry["posted_date"] = timestamp
    tip_entry["image_path"] = image_path
    json_data.append(tip_entry)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Tip entry and image path logged to JSON.")

def main():
    search_tool = DuckDuckGoSearchRun()

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=AVALAI_API_KEY,
        base_url="https://api.avalai.ir/v1",
        temperature=0.7
    )

    llm_with_tools = llm.bind_tools([search_tool])

    print("--- Step 1: Searching Web for Grounded Knowledge ---")
    query = "latest agentic AI architecture design patterns 2026"
    
    messages = [
        SystemMessage(content="You are an expert AI Agents Educator. Search for recent insights, then propose tips."),
        HumanMessage(content=f"Search query: '{query}'")
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_output = search_tool.invoke(tool_call['args'])
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call['id']))

    print("\n--- Step 2: Generating Candidate Tips ---")
    structured_llm = llm.with_structured_output(TipCandidates)
    messages.append(HumanMessage(content="Generate 3 to 5 candidates grounded in search results."))

    result: TipCandidates = structured_llm.invoke(messages)

    print("\n--- Step 3: Vector Deduplication (ChromaDB < 0.85) ---")
    surviving_candidates: List[AgentTip] = []

    for idx, tip in enumerate(result.tips, 1):
        duplicate, score = is_duplicate(tip, threshold=0.85)
        if duplicate:
            print(f"❌ Candidate #{idx} Dropped (Similarity {score:.2f} >= 0.85)")
        else:
            print(f"✨ Candidate #{idx} Passed (Similarity {score:.2f} < 0.85)")
            surviving_candidates.append(tip)

    print("\n--- Step 4: Selecting Best Candidate ---")
    best_tip = select_best_candidate(surviving_candidates)

    if best_tip:
        # Step 5: تولید تصویر اختصاصی برای نکته انتخاب شده
        image_path = generate_and_save_image(best_tip)

        # Step 6: ذخیره نهایی اطلاعات و مسیر تصویر
        save_posted_tip(best_tip, image_path)
    else:
        print("⚠️ No surviving unique candidates found to post today.")

if __name__ == "__main__":
    main()