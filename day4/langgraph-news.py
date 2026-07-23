import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------
# 1. 환경변수 로드
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ---------------------------------------------------------
# 2. State 정의
# ---------------------------------------------------------
class AgentState(TypedDict):
    keyword: str        # 검색 키워드
    news: List[str]     # 검색된 뉴스 목록
    summary: str        # 요약 결과
    is_done: bool       # 완료 여부


# ---------------------------------------------------------
# 3. 노드 정의
# ---------------------------------------------------------
def search_node(state: AgentState) -> AgentState:
    """키워드로 뉴스를 검색한다. (현재는 임시 더미 데이터 반환)"""
    keyword = state["keyword"]
    print(f"[search_node] '{keyword}' 뉴스 검색 중...")

    # TODO: 실제 검색 API(Naver, Tavily, SerpAPI 등)로 교체
    dummy_db = {
        "AI": [
            "오픈AI, 차세대 추론 모델 공개... 수학·코딩 성능 대폭 향상",
            "국내 기업 70%가 생성형 AI 도입 검토, 실제 적용은 20% 그쳐",
            "EU AI Act 본격 시행, 고위험 AI 시스템 규제 강화",
        ],
        "반도체": [
            "HBM4 양산 경쟁 본격화, 메모리 3사 투자 확대",
            "미국 반도체 보조금 집행 지연에 업계 우려 확산",
        ],
    }

    news = dummy_db.get(keyword, [])
    if not news and keyword:
        # 키워드가 사전에 없으면 빈 리스트 -> end_node로 분기됨
        news = []

    print(f"[search_node] {len(news)}건 검색 완료")
    return {"news": news}


def summarize_node(state: AgentState) -> AgentState:
    """검색된 뉴스를 LLM으로 요약한다."""
    keyword = state["keyword"]
    news = state["news"]
    print("[summarize_node] 요약 생성 중...")

    news_text = "\n".join(f"- {item}" for item in news)

    messages = [
        SystemMessage(content=(
            "너는 뉴스 요약 전문가다. 주어진 뉴스 목록을 한국어로 "
            "3~4문장으로 핵심만 간결하게 요약하라. "
            "추측이나 없는 내용은 덧붙이지 마라."
        )),
        HumanMessage(content=f"키워드: {keyword}\n\n뉴스 목록:\n{news_text}"),
    ]

    response = llm.invoke(messages)
    return {"summary": response.content}


def end_node(state: AgentState) -> AgentState:
    """작업 완료를 표시한다."""
    print("[end_node] 작업 종료")

    summary = state.get("summary") or "검색된 뉴스가 없어 요약할 내용이 없습니다."
    return {"summary": summary, "is_done": True}


# ---------------------------------------------------------
# 4. 조건 분기 함수
# ---------------------------------------------------------
def should_summarize(state: AgentState) -> str:
    """뉴스가 있으면 summarize, 없으면 end로 분기"""
    if state.get("news"):
        return "summarize"
    return "end"


# ---------------------------------------------------------
# 5. 그래프 구성
# ---------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("end_node", end_node)

    workflow.add_edge(START, "search")

    workflow.add_conditional_edges(
        "search",
        should_summarize,
        {
            "summarize": "summarize",
            "end": "end_node",
        },
    )

    workflow.add_edge("summarize", "end_node")
    workflow.add_edge("end_node", END)

    return workflow.compile()


# ---------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------
# ---------------------------------------------------------
# 6. 실행 (반복 모드)
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    EXIT_WORDS = {"종료", "exit", "quit", "q"}

    print("=" * 50)
    print("뉴스 검색·요약 에이전트")
    print("종료하려면 '종료' 를 입력하세요.")
    print("=" * 50)

    while True:
        keyword = input("\n검색할 키워드를 입력하세요: ").strip()

        if keyword in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break

        if not keyword:
            print("키워드를 입력해 주세요.")
            continue

        initial_state: AgentState = {
            "keyword": keyword,
            "news": [],
            "summary": "",
            "is_done": False,
        }

        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        print("\n" + "=" * 50)
        print(f"키워드   : {result['keyword']}")
        print(f"검색건수 : {len(result['news'])}건")
        print(f"완료여부 : {result['is_done']}")
        print("-" * 50)
        print("[요약 결과]")
        print(result["summary"])
        print("=" * 50)