import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL = os.getenv("TELEGRAM_CHANNEL")

if not BOT_TOKEN or not CHANNEL:
    raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL missing in .env")

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {
    "chat_id": CHANNEL,
    "text": "hello"
}

response = requests.post(url, json=payload)
print(response.status_code, response.text)
