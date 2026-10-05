import os
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
BASE_URL = "https://api.avalai.ir/v1"

if not AVALAI_API_KEY:
    raise ValueError("AVALAI_API_KEY missing in .env")

headers = {"Authorization": f"Bearer {AVALAI_API_KEY}"}

models_url = BASE_URL + "/models"
models_resp = requests.get(models_url, headers=headers)
print("GET /models status:", models_resp.status_code)
print(models_resp.text)

class AgentTip(BaseModel):
    title_en: str = Field(description="Short title in English.")
    body_en: str = Field(description="Detailed explanation in English.")
    title_fa: str = Field(description="Natural, fluent Persian title written the way a Persian tech educator would phrase it. Avoid literal word-for-word translation.")
    body_fa: str = Field(description="Natural, fluent Persian body explanation written the way a Persian tech educator would phrase it. Avoid literal word-for-word translation.")
    difficulty: str = Field(description="Difficulty level: Beginner, Intermediate, or Advanced.")
    source_url: str = Field(default="", description="Source URL or document link.")

model_id = "deepseek-chat"
llm = ChatOpenAI(model=model_id, api_key=AVALAI_API_KEY, base_url=BASE_URL, temperature=0.7)
structured_llm = llm.with_structured_output(AgentTip)

response = structured_llm.invoke([
    SystemMessage(content="You are an AI agents educator. Generate one short tip about agentic AI. Provide natural Persian translations for title and body, not literal translations."),
    HumanMessage(content="Generate ONE short tip about agentic AI in English, with Persian translations.")
])

print("Structured tip:", response.model_dump())
