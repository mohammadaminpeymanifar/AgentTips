import os
from dotenv import load_dotenv
from openai import OpenAI

# بارگذاری متغیرهای محیطی
load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")

if not AVALAI_API_KEY:
    raise ValueError("کلید AVALAI_API_KEY در فایل .env پیدا نشد.")

# مقداردهی اولیه کلاینت OpenAI با آدرس اختصاصی AvalAI
client = OpenAI(
    api_key=AVALAI_API_KEY,
    base_url="https://api.avalai.ir/v1"
)

def main():
    # ۱. دریافت لیست مدل‌های موجود در AvalAI
    print("--- Fetching available models from AvalAI ---")
    try:
        models = client.models.list()
        deepseek_models = []
        
        for model in models.data:
            print(f"- {model.id}")
            if "deepseek" in model.id.lower():
                deepseek_models.append(model.id)
                
        print("\nFound DeepSeek models:", deepseek_models)
        
        # انتخاب اولین مدل DeepSeek یافت شده یا مدل پیش‌فرض
        selected_model = deepseek_models[0] if deepseek_models else "deepseek-chat"
        print(f"\n---> Using model: {selected_model}\n")

    except Exception as e:
        print(f"Error fetching models: {e}")
        return

    # ۲. فراخوانی مدل جهت دریافت یک نکته کوتاه در نقش مدرس AI Agent
    print("--- Generating Agentic AI Tip ---")
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI agents educator. Provide clear, highly practical, and concise insights about agentic AI."
            },
            {
                "role": "user",
                "content": "Generate ONE short, impactful tip about agentic AI in English."
            }
        ],
        temperature=0.7,
        max_tokens=150
    )

    tip = response.choices[0].message.content
    print("\nResult:")
    print("--------------------------------------------------")
    print(tip)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()