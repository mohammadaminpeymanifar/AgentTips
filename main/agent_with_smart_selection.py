import os
import json
from datetime import datetime
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import chromadb
from chromadb.utils import embedding_functions

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# بارگذاری متغیرهای محیطی
load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
if not AVALAI_API_KEY:
    raise ValueError("کلید AVALAI_API_KEY در فایل .env پیدا نشد.")

# مسیرهای داده
DATA_DIR = "./data"
JSON_PATH = os.path.join(DATA_DIR, "posted_tips.json")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)

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

def get_last_two_difficulties() -> List[str]:
    """خواندن سطح سختی دو پست اخیر از فایل JSON"""
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data:
                return []
            # استخراج سطح سختی آخرین موارد ذخیره‌شده
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
    """
    انتخاب بهترین کاندیدا:
    1. اولویت به سطوحی که با ۲ پست اخیر متفاوت هستند.
    2. حفظ رتبه‌بندی/تناسب اولیه مدل (Highest Relevancy First).
    """
    if not candidates:
        return None

    recent_difficulties = get_last_two_difficulties()
    print(f"\n📊 Recent 2 posted difficulties: {recent_difficulties}")

    # تفکیک کاندیداها به دو دسته: سطوح جدید و سطوح تکراری
    diverse_candidates = [
        c for c in candidates if c.difficulty not in recent_difficulties
    ]

    if diverse_candidates:
        selected = diverse_candidates[0]
        print(f"🎯 Selected candidate with DIVERSE difficulty ({selected.difficulty}): '{selected.title_en}'")
        return selected

    # اگر تمام کاندیداها هم‌سطح ۲ پست اخیر بودند، اولین (مرتبط‌ترین) مورد انتخاب می‌شود
    selected = candidates[0]
    print(f"⚠️ Fallback to top-relevant candidate ({selected.difficulty}): '{selected.title_en}'")
    return selected

def save_posted_tip(tip: AgentTip):
    """ذخیره در ChromaDB و JSON"""
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
    json_data.append(tip_entry)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully saved tip to memory & JSON!")

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
    messages.append(HumanMessage(content="Generate 3 to 5 candidates grounded in search results, ordered by relevance."))

    result: TipCandidates = structured_llm.invoke(messages)

    print("\n--- Step 3: Vector Deduplication (ChromaDB Similarity < 0.85) ---")
    surviving_candidates: List[AgentTip] = []

    for idx, tip in enumerate(result.tips, 1):
        duplicate, score = is_duplicate(tip, threshold=0.85)
        if duplicate:
            print(f"❌ Candidate #{idx} Dropped (Similarity {score:.2f} >= 0.85): '{tip.title_en}'")
        else:
            print(f"✨ Candidate #{idx} Passed (Max Similarity {score:.2f} < 0.85): '{tip.title_en}'")
            surviving_candidates.append(tip)

    print("\n--- Step 4: Selecting Single Best Candidate ---")
    best_tip = select_best_candidate(surviving_candidates)

    if best_tip:
        print("\n🏆 WINNING CANDIDATE DETAILS:")
        print("=" * 60)
        print(f"🇬🇧 Title (EN): {best_tip.title_en}")
        print(f"📝 Body (EN):  {best_tip.body_en}")
        print(f"🇮🇷 Title (FA): {best_tip.title_fa}")
        print(f"📝 Body (FA):  {best_tip.body_fa}")
        print(f"📊 Difficulty: {best_tip.difficulty}")
        if best_tip.source_url:
            print(f"🔗 Source:     {best_tip.source_url}")
        print("=" * 60)

        # ذخیره‌سازی در حافظه
        save_posted_tip(best_tip)
    else:
        print("⚠️ No surviving unique candidates found to post today.")

if __name__ == "__main__":
    main()