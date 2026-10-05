import os
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# بارگذاری متغیرها
load_dotenv()

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
if not AVALAI_API_KEY:
    raise ValueError("کلید AVALAI_API_KEY در فایل .env پیدا نشد.")

# ۱. مدل تک‌نکته
class AgentTip(BaseModel):
    title_en: str = Field(
        description="Short, catchy title for the tip in English."
    )
    body_en: str = Field(
        description="The detailed body/explanation of the tip in English."
    )
    title_fa: str = Field(
        description="A natural, fluent Persian title like an experienced Persian tech educator speaking naturally."
    )
    body_fa: str = Field(
        description="A natural, fluent Persian body explanation. Explain idiomatic tech concepts naturally, avoid word-for-word translation."
    )
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        description="Difficulty level of the concept."
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Source URL or document link found during web search that grounds this tip."
    )

# ۲. مدل کاندیداها (بین ۳ تا ۵ نکته)
class TipCandidates(BaseModel):
    tips: List[AgentTip] = Field(
        description="A list of 3 to 5 distinct agentic AI tip candidates grounded in the web search results.",
        min_items=3,
        max_items=5
    )

def main():
    # ساخت ابزار جستجو
    search_tool = DuckDuckGoSearchRun()

    # تعریف مدل با قابلیت Tool Calling
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=AVALAI_API_KEY,
        base_url="https://api.avalai.ir/v1",
        temperature=0.7
    )

    # اتصال ابزار جستجو به مدل
    llm_with_tools = llm.bind_tools([search_tool])

    print("--- Step 1: Agent Searching Web for Recent Agentic AI Insights ---")
    query = "recent advancements trends architectural patterns agentic AI 2026"
    
    system_prompt = (
        "You are an expert AI Agents Educator. "
        "First, use the provided search tool to find real, recent insights about Agentic AI. "
        "After getting the search results, synthesize them into 3 to 5 candidate tips."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Search for recent trends using this query: '{query}' and prepare insights.")
    ]

    # فراخوانی اول جهت گرفتن تصمیم Tool Call
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # اگر مدل تصمیم گرفت جستجو انجام دهد:
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔍 Executing Tool: {tool_call['name']} with query: {tool_call['args']}")
            tool_output = search_tool.invoke(tool_call['args'])
            
            # اضافه کردن پاسخ ابزار به تاریخچه پیام‌ها
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call['id']
                )
            )

    print("\n--- Step 2: Generating 3-5 Candidate Tips with Structured Output ---")
    
    # ساخت Structured Output بر اساس مدل TipCandidates
    structured_llm = llm.with_structured_output(TipCandidates)

    final_prompt = (
        "Based ON THE SEARCH RESULTS ABOVE, generate between 3 to 5 high-impact, actionable candidate tips. "
        "Ensure all Persian fields sound completely natural, fluent, and educator-like."
    )
    messages.append(HumanMessage(content=final_prompt))

    # فراخوانی نهایی با ساختار Pydantic
    result: TipCandidates = structured_llm.invoke(messages)

    print(f"\nSuccessfully generated {len(result.tips)} candidate tips:\n")
    print("=" * 60)
    for idx, tip in enumerate(result.tips, 1):
        print(f"Candidate #{idx} [{tip.difficulty}]")
        print(f"🇬🇧 EN: {tip.title_en}")
        print(f"    {tip.body_en}")
        print(f"🇮🇷 FA: {tip.title_fa}")
        print(f"    {tip.body_fa}")
        if tip.source_url:
            print(f"🔗 Source: {tip.source_url}")
        print("-" * 60)

if __name__ == "__main__":
    main()