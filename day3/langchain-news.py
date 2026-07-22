# assistant.py
import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


@tool
def search_news(query: str) -> str:
    """최신 뉴스를 검색한다. 뉴스, 기사, 오늘의 소식 등을 물어볼 때 사용한다."""
    url = "https://newsdata.io/api/1/news"
    params = {"apikey": NEWSDATA_API_KEY, "q": query, "language": "ko", "size": 5}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        articles = res.json().get("results", [])
        if not articles:
            return f"'{query}'에 대한 뉴스를 찾지 못했습니다."
        return "\n".join(
            f"- {a.get('title')} ({a.get('source_id')})\n  {a.get('link')}"
            for a in articles
        )
    except Exception as e:
        return f"뉴스 검색 실패: {e}"


@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 조회한다. 날씨, 기온, 비 여부를 물어볼 때 사용한다."""
    return f"{city}의 현재 날씨는 맑음, 기온 23도, 습도 45%입니다. (임시 데이터)"


tools = [search_news, get_weather]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "당신은 친절한 한국어 AI 어시스턴트입니다. "
        "필요할 때 제공된 도구를 사용해 정확한 정보를 답변하세요."
    ),
)


def main():
    chat_history = []
    print("AI 어시스턴트를 시작합니다. ('종료' 입력 시 프로그램 종료)")
    while True:
        user_input = input("\n사용자: ").strip()
        if user_input == "종료":
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            continue
        try:
            chat_history.append(HumanMessage(content=user_input))
            result = agent.invoke({"messages": chat_history})

            # verbose 대체: Tool 선택 과정 출력
            for msg in result["messages"][len(chat_history) - 1:]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[Tool 호출] {tc['name']}({tc['args']})")
                elif msg.__class__.__name__ == "ToolMessage":
                    print(f"[Tool 결과] {msg.content[:200]}")

            chat_history = result["messages"]
            print(f"\n어시스턴트: {chat_history[-1].content}")
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()