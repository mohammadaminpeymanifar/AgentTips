import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

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

client = OpenAI(api_key=AVALAI_API_KEY, base_url=BASE_URL)

model_id = "deepseek-chat"
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are an AI agents educator. Write one short, practical tip about agentic AI."},
        {"role": "user", "content": "Generate ONE short tip about agentic AI in English."}
    ]
)
print("Model response:", response.choices[0].message.content)
