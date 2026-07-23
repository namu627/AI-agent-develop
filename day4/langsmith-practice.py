import os
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from ddgs import DDGS

# ---------------------------------------------------------
# 1. 환경변수 로드
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "react-news-agent")
    print("[init] LangSmith 추적 활성화")
else:
    print("[init] LangSmith 키 없음 - 추적 비활성화")

NO_RESULT = "NO_RESULT"   # 검색 결과 없음을 나타내는 고정 토큰


# ---------------------------------------------------------
# 2. Tool 정의 (실제 검색)
# ---------------------------------------------------------
@tool
def search_news(keyword: str) -> str:
    """주어진 키워드로 최신 뉴스를 검색해 실제 기사 목록을 반환한다.
    검색된 기사가 없으면 NO_RESULT 를 반환한다."""
    print(f"    >> [Tool] search_news(keyword='{keyword}')")

    kw = keyword.strip()
    if not kw:
        return NO_RESULT

    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(kw, region="kr-kr", max_results=5))
    except Exception as e:
        print(f"    >> [Tool] 검색 오류: {e}")
        return f"SEARCH_ERROR: {e}"

    if not results:
        print("    >> [Tool] 0건")
        return NO_RESULT

    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip().replace("\n", " ")[:150]
        source = (r.get("source") or "").strip()
        date = (r.get("date") or "")[:10]
        lines.append(f"{i}. [{source} / {date}] {title}\n   {body}")

    print(f"    >> [Tool] {len(results)}건 검색 완료")
    return "\n".join(lines)


TOOLS = [search_news]
TOOL_MAP = {t.name: t for t in TOOLS}


# ---------------------------------------------------------
# 3. State 정의
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_action: str      # "tool" | "end"
    tool_result: str      # 마지막 도구 실행 결과


# ---------------------------------------------------------
# 4. LLM
# ---------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = f"""너는 뉴스 리서치 어시스턴트다. ReAct 방식으로 동작한다.

[행동 규칙]
1. 사용자가 주제를 물으면 먼저 search_news 도구로 뉴스를 검색한다.
2. 검색 결과를 받으면 도구를 다시 호출하지 말고 최종 답변을 작성한다.
3. 답변은 한국어로 3~5문장이며, 마지막에 참고 기사 제목을 나열한다.

[환각 금지 - 반드시 지킬 것]
- 도구가 반환한 기사 본문에 실제로 있는 내용만 사용한다.
- 도구 결과가 "{NO_RESULT}" 이면 요약을 만들어내지 말고 정확히 이렇게만 답한다:
  "'<키워드>'에 대한 뉴스 검색 결과가 없습니다. 다른 키워드로 검색해 보세요."
- 도구 결과가 "SEARCH_ERROR"로 시작하면 검색에 실패했다고 알리고 재시도를 안내한다.
- 시장 규모, 성장률, 투자 확대, 규제 정비 같은 일반론을 근거 없이 덧붙이지 않는다.
- 기사에 없는 수치·전망·인용은 절대 만들어내지 않는다.
"""


# ---------------------------------------------------------
# 5. 노드 정의
# ---------------------------------------------------------
def think_node(state: AgentState) -> AgentState:
    """LLM이 상황을 판단하고 다음 행동을 결정한다."""
    print("[think_node] 판단 중...")

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response: AIMessage = llm_with_tools.invoke(messages)

    if response.tool_calls:
        next_action = "tool"
        for tc in response.tool_calls:
            print(f"    -> 도구 호출 결정: {tc['name']}({tc['args']})")
    else:
        next_action = "end"
        print("    -> 최종 답변 생성 결정")

    return {"messages": [response], "next_action": next_action}


def act_node(state: AgentState) -> AgentState:
    """think_node가 요청한 도구를 실행한다."""
    print("[act_node] 도구 실행")

    last_msg = state["messages"][-1]
    tool_messages = []
    last_result = ""

    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if name in TOOL_MAP:
            try:
                result = TOOL_MAP[name].invoke(args)
            except Exception as e:
                result = f"SEARCH_ERROR: {e}"
        else:
            result = f"SEARCH_ERROR: 알 수 없는 도구 {name}"

        last_result = str(result)
        tool_messages.append(
            ToolMessage(content=last_result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages, "tool_result": last_result}


# ---------------------------------------------------------
# 6. 조건 분기
# ---------------------------------------------------------
def route(state: AgentState) -> str:
    return "tool" if state["next_action"] == "tool" else "end"


# ---------------------------------------------------------
# 7. 그래프 구성
# ---------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("think", think_node)
    workflow.add_node("act", act_node)

    workflow.add_edge(START, "think")
    workflow.add_conditional_edges(
        "think", route, {"tool": "act", "end": END}
    )
    workflow.add_edge("act", "think")

    return workflow.compile()


# ---------------------------------------------------------
# 8. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    EXIT_WORDS = {"종료", "exit", "quit", "q"}

    print("=" * 60)
    print("ReAct 뉴스 리서치 에이전트 (실시간 검색)")
    print("종료하려면 '종료' 를 입력하세요.")
    print("=" * 60)

    while True:
        keyword = input("\n검색할 키워드를 입력하세요: ").strip()

        if keyword in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break
        if not keyword:
            print("키워드를 입력해 주세요.")
            continue

        initial_state: AgentState = {
            "messages": [HumanMessage(content=f"{keyword}에 대한 최신 뉴스를 알려줘.")],
            "next_action": "",
            "tool_result": "",
        }

        print("-" * 60)
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 10})
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        print("-" * 60)
        print("[최종 답변]")
        print(result["messages"][-1].content)
        print("=" * 60)