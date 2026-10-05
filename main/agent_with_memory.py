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

# بارگذاری متغیرها
load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
if not AVALAI_API_KEY:
    raise ValueError("کلید AVALAI_API_KEY در فایل .env پیدا نشد.")

# مسیرهای دیتابیس و لاگ
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

# ۲. مدیریت دیتابیس برداری Chroma
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="posted_tips",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # محاسبه فاصله/شباهت کسینوسی
)

def is_duplicate(tip: AgentTip, threshold: float = 0.85) -> tuple[bool, float]:
    """بررسی میزان شباهت کسینوسی با نکات قبلی ذخیره‌شده در ChromaDB"""
    if collection.count() == 0:
        return False, 0.0

    text_to_check = f"{tip.title_en} {tip.body_en}"
    results = collection.query(
        query_texts=[text_to_check],
        n_results=1
    )

    if results and results["distances"] and results["distances"][0]:
        # در Chroma با معیارهای Cosine، مقدار distance به شکل (1 - cosine_similarity) است.
        distance = results["distances"][0][0]
        similarity = 1.0 - distance
        
        if similarity >= threshold:
            return True, similarity
        return False, similarity

    return False, 0.0

def save_posted_tip(tip: AgentTip):
    """ذخیره نکته جدید در ChromaDB و فایل JSON"""
    timestamp = datetime.now().isoformat()
    doc_text = f"{tip.title_en} {tip.body_en}"
    doc_id = f"tip_{int(datetime.now().timestamp())}"

    # ۱. ذخیره در ChromaDB
    collection.add(
        documents=[doc_text],
        metadatas=[{"title_en": tip.title_en, "date": timestamp}],
        ids=[doc_id]
    )

    # ۲. ذخیره در JSON
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

    print(f"✅ Tip saved to ChromaDB and {JSON_PATH}")

def main():
    search_tool = DuckDuckGoSearchRun()

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=AVALAI_API_KEY,
        base_url="https://api.avalai.ir/v1",
        temperature=0.7
    )

    llm_with_tools = llm.bind_tools([search_tool])

    print("--- Step 1: Agent Searching Web ---")
    query = "recent agentic AI architectures trends 2026"
    
    messages = [
        SystemMessage(content="You are an expert AI Agents Educator. Search for recent insights, then propose tips."),
        HumanMessage(content=f"Search query: '{query}'")
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔍 Searching: {tool_call['args']}")
            tool_output = search_tool.invoke(tool_call['args'])
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call['id']))

    print("\n--- Step 2: Generating Candidates ---")
    structured_llm = llm.with_structured_output(TipCandidates)
    messages.append(HumanMessage(content="Generate 3 to 5 candidates grounded in search results."))

    result: TipCandidates = structured_llm.invoke(messages)

    print("\n--- Step 3: Filtering Candidates via ChromaDB Vector Similarity ---")
    unique_candidates: List[AgentTip] = []

    for idx, tip in enumerate(result.tips, 1):
        duplicate, score = is_duplicate(tip, threshold=0.85)
        if duplicate:
            print(f"❌ Candidate #{idx} dropped (Similarity: {score:.2f} >= 0.85): '{tip.title_en}'")
        else:
            print(f"✨ Candidate #{idx} accepted (Max Similarity: {score:.2f} < 0.85): '{tip.title_en}'")
            unique_candidates.append(tip)

    print(f"\nFinal Unique Candidates Count: {len(unique_candidates)}")

    # اگر حداقل یک کاندیدای غیرتکراری وجود داشت، اولین مورد به عنوان نمونه ذخیره می‌شود
    if unique_candidates:
        selected_tip = unique_candidates[0]
        print("\n--- Saving Selected Tip to Memory ---")
        save_posted_tip(selected_tip)

if __name__ == "__main__":
    main()