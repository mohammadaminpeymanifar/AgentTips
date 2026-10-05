import os
import urllib.parse
import urllib.request
import urllib.error
from dotenv import load_dotenv

# بارگذاری متغیرها از فایل .env
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL = os.getenv("TELEGRAM_CHANNEL")

def send_message():
    if not BOT_TOKEN or not CHANNEL:
        raise ValueError("متغیرهای TELEGRAM_BOT_TOKEN یا TELEGRAM_CHANNEL یافت نشدند.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": CHANNEL,
        "text": "hello"
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload)
    
    try:
        with urllib.request.urlopen(req) as response:
            print("پاسخ تلگرام:", response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"خطای تلگرام ({e.code}): {error_body}")

if __name__ == "__main__":
    send_message()