# 🤖 TipsAgent - Automated Bilingual AI Educator

TipsAgent is an autonomous AI Agent system that researches, generates, deduplicates, illustrates, and posts high-quality bilingual (English/Persian) tips about Agentic AI directly to a Telegram channel.

---

## 🌟 Key Features

* **Grounded Knowledge:** Performs real-time web search (`DuckDuckGoSearchRun`) before generating tips.
* **Structured Output:** Enforces strict Pydantic schema validation via LangChain & AvalAI (DeepSeek).
* **Bilingual Content:** Native English and fluent, idiomatic Persian adapted for technical audiences.
* **Vector Deduplication:** Uses local `ChromaDB` cosine similarity (0.85 threshold) to prevent repeated concepts.
* **Smart Diversity:** Selects candidates based on difficulty level diversity relative to recent posts.
* **AI Visual Metaphors:** Dynamically generates isometric 2.5D sci-fi illustrations via DALL-E 3 API.
* **Resilient Architecture:** Automatic exponential retries via `tenacity` on external API failures.

---

## 🛠️ Quick Start Setup

### 1. Clone & Setup Environment
```bash
git clone <your-repo-url>
cd tips-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt