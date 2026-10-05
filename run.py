import os
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import chromadb
from chromadb.utils import embedding_functions

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TipsAgent")

# Load environment variables
load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL = os.getenv("TELEGRAM_CHANNEL")

if not AVALAI_API_KEY:
    raise ValueError("Missing AVALAI_API_KEY in .env file.")
if not BOT_TOKEN or not CHANNEL:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL in .env file.")

DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSON_PATH = os.path.join(DATA_DIR, "posted_tips.json")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Database & API Clients
# ---------------------------------------------------------
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="posted_tips",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

openai_client = OpenAI(
    api_key=AVALAI_API_KEY,
    base_url="https://api.avalai.ir/v1"
)

# ---------------------------------------------------------
# Retryable External Calls
# ---------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def fetch_search_results(search_tool: DuckDuckGoSearchRun, query: str) -> str:
    logger.info(f"[SEARCH] Executing DuckDuckGo query: '{query}'")
    return search_tool.invoke(query)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_candidates_with_llm(structured_llm, messages) -> TipCandidates:
    logger.info("[LLM] Calling AvalAI LLM to generate structured candidate tips...")
    return structured_llm.invoke(messages)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_image_api(prompt: str):
    logger.info("[IMAGE] Requesting image generation from AvalAI Image Endpoint...")
    return openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def send_telegram_photo(url: str, request_data: bytes, boundary: str):
    logger.info("[TELEGRAM] Sending photo & caption via Telegram Bot API...")
    req = urllib.request.Request(
        url,
        data=request_data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")

# ---------------------------------------------------------
# Logic Functions
# ---------------------------------------------------------

def get_last_two_difficulties() -> List[str]:
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data:
                return []
            return [t.get("difficulty") for t in data[-2:] if "difficulty" in t]
    except Exception as e:
        logger.warning(f"Could not read last difficulties: {e}")
        return []

def is_duplicate(tip: AgentTip, threshold: float = 0.85) -> tuple[bool, float]:
    if collection.count() == 0:
        return False, 0.0

    text_to_check = f"{tip.title_en} {tip.body_en}"
    results = collection.query(query_texts=[text_to_check], n_results=1)

    if results and results["distances"] and results["distances"][0]:
        distance = results["distances"][0][0]
        similarity = 1.0 - distance
        return similarity >= threshold, similarity

    return False, 0.0

def select_best_candidate(candidates: List[AgentTip]) -> Optional[AgentTip]:
    if not candidates:
        return None

    recent_difficulties = get_last_two_difficulties()
    logger.info(f"[DIVERSITY] Recent difficulties in memory: {recent_difficulties}")

    diverse = [c for c in candidates if c.difficulty not in recent_difficulties]
    if diverse:
        logger.info(f"[SELECT] Picked candidate with DIVERSE difficulty ({diverse[0].difficulty}): '{diverse[0].title_en}'")
        return diverse[0]

    logger.info(f"[SELECT] Fallback to top-relevant candidate ({candidates[0].difficulty}): '{candidates[0].title_en}'")
    return candidates[0]

def create_and_download_image(tip: AgentTip) -> str:
    image_prompt = (
        f"Futuristic sci-fi editorial illustration representing the tech concept: '{tip.title_en}'. "
        f"Specific focal metaphor: {tip.body_en}. "
        "Clean 2.5D isometric-leaning style, soft volumetric lighting, deep layered planes, "
        "futuristic translucent materials, vibrant glowing accents. "
        "Educational visual metaphor, single clear focal subject. "
        "NO mascot robot, NO human face, NO text, NO letters, NO words, high resolution, 3D render feel."
    )

    response = generate_image_api(image_prompt)
    image_url = response.data[0].url

    timestamp = int(datetime.now().timestamp())
    image_path = os.path.join(IMAGES_DIR, f"tip_{timestamp}.png")

    logger.info(f"[IMAGE] Downloading generated image to {image_path}...")
    urllib.request.urlretrieve(image_url, image_path)
    return image_path

def post_to_telegram(tip: AgentTip, image_path: str):
    caption = (
        f"<b>{tip.title_en}</b>\n"
        f"{tip.body_en}\n\n"
        f"<b>{tip.title_fa}</b>\n"
        f"{tip.body_fa}\n\n"
        f"See you tomorrow for another AI agents tip!\n"
        f"{CHANNEL}"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    body = bytearray()
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{CHANNEL}\r\n".encode("utf-8"))
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{os.path.basename(image_path)}\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8"))
    body.extend(image_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    res = send_telegram_photo(url, bytes(body), boundary)
    logger.info("[TELEGRAM] Message posted successfully!")

def save_posted_tip(tip: AgentTip, image_path: str):
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

    logger.info("[MEMORY] Saved tip to ChromaDB vector store and JSON log.")

# ---------------------------------------------------------
# Pipeline Entrypoint
# ---------------------------------------------------------

def run_pipeline():
    logger.info("=== STARTING TIPS AGENT PIPELINE ===")

    try:
        search_tool = DuckDuckGoSearchRun()
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=AVALAI_API_KEY,
            base_url="https://api.avalai.ir/v1",
            temperature=0.7
        )

        # Step 1: Web Search
        query = "latest agentic AI architecture design patterns 2026"
        search_results = fetch_search_results(search_tool, query)

        # Step 2: Generate Candidates
        system_prompt = "You are an expert AI Agents Educator. Ground your insights on search results."
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Search findings:\n{search_results}\n\nGenerate 3 to 5 candidate tips.")
        ]
        structured_llm = llm.with_structured_output(TipCandidates)
        result: TipCandidates = generate_candidates_with_llm(structured_llm, messages)
        logger.info(f"[GENERATION] Generated {len(result.tips)} raw candidates.")

        # Step 3: Vector Deduplication
        surviving_candidates = []
        for idx, tip in enumerate(result.tips, 1):
            duplicate, score = is_duplicate(tip, threshold=0.85)
            if duplicate:
                logger.info(f"[DEDUP] Candidate #{idx} DROPPED (Similarity: {score:.2f} >= 0.85): '{tip.title_en}'")
            else:
                logger.info(f"[DEDUP] Candidate #{idx} PASSED (Similarity: {score:.2f} < 0.85): '{tip.title_en}'")
                surviving_candidates.append(tip)

        # Step 4: Selection
        best_tip = select_best_candidate(surviving_candidates)

        if not best_tip:
            logger.warning("[PIPELINE] No unique candidates survived deduplication. Skipping post today.")
            return

        # Step 5: Image Generation
        image_path = create_and_download_image(best_tip)

        # Step 6: Post to Telegram
        post_to_telegram(best_tip, image_path)

        # Step 7: Memory Logging
        save_posted_tip(best_tip, image_path)

        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        logger.error(f"[PIPELINE FAILED] Error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()