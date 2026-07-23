import os
import glob
from datetime import datetime
from typing import TypedDict, List

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------
# 1. 환경변수 로드
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

DOCS_DIR = "docs"
JOB_DOC_FILES = ["job_posting.txt", "company_culture.txt", "interview_faq.txt", "tech_stack.txt"]
RESUME_FILE = os.path.join(DOCS_DIR, "자소서.txt")
RESULTS_DIR = "results"
TOP_K = 4

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ---------------------------------------------------------
# 2. 뉴스 검색 Tool (NewsData.io 실제 연동)
# ---------------------------------------------------------
@tool
def search_job_news(query: str) -> str:
    """채용, 취업, 산업 동향 관련 최신 뉴스를 검색한다. query는 검색 키워드."""
    if not NEWSDATA_API_KEY:
        return "NEWSDATA_API_KEY가 설정되어 있지 않아 뉴스를 검색할 수 없습니다."

    try:
        response = requests.get(
            "https://newsdata.io/api/1/latest",
            params={"apikey": NEWSDATA_API_KEY, "q": query, "language": "ko"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"뉴스 검색 중 오류가 발생했습니다: {e}"

    articles = data.get("results") or []
    if not articles:
        return f"'{query}'에 대한 뉴스를 찾지 못했습니다."

    lines = []
    for i, article in enumerate(articles[:5], 1):
        title = article.get("title", "제목 없음")
        pub_date = article.get("pubDate", "날짜 미상")
        desc = (article.get("description") or "").strip()
        link = article.get("link", "")
        lines.append(f"{i}. {title} ({pub_date})\n   요약: {desc[:150]}\n   링크: {link}")

    return "\n".join(lines)


# ---------------------------------------------------------
# 3. RAG 파이프라인 (채용공고 문서 실제 연동)
# ---------------------------------------------------------
def load_job_documents() -> List[Document]:
    documents: List[Document] = []
    for filename in JOB_DOC_FILES:
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(path):
            print(f"[setup] 경고: '{path}' 파일을 찾을 수 없어 건너뜁니다.")
            continue

        try:
            loaded = TextLoader(path, encoding="utf-8").load()
        except UnicodeDecodeError:
            loaded = TextLoader(path, encoding="cp949").load()

        for d in loaded:
            d.metadata["source_file"] = filename
        documents.extend(loaded)
        print(f"  - {filename} 로드 완료")

    return documents


def build_job_vectorstore() -> FAISS:
    print("[setup] 채용 관련 문서 로드 중...")
    documents = load_job_documents()
    if not documents:
        raise FileNotFoundError(f"'{DOCS_DIR}' 폴더에서 채용 관련 문서를 찾지 못했습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[setup] 청크 분할 완료: {len(chunks)}개")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[setup] FAISS 벡터 저장소 생성 완료\n")
    return vectorstore


job_vectorstore = build_job_vectorstore()
job_retriever = job_vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", "unknown")
        parts.append(f"[발췌 {i} | 출처: {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------
# 4. State 정의
# ---------------------------------------------------------
class SupervisorState(TypedDict):
    user_input: str
    next_agent: str
    news_result: str
    rag_result: str
    resume_result: str
    final_answer: str


AGENT_LABELS = {
    "news": "📰 뉴스 검색 에이전트",
    "rag": "📄 채용공고 분석(RAG) 에이전트",
    "resume": "✍️ 자소서 피드백 에이전트",
}


# ---------------------------------------------------------
# 5. 노드 정의
# ---------------------------------------------------------
def supervisor_node(state: SupervisorState) -> SupervisorState:
    user_input = state["user_input"]
    print("[supervisor_node] 요청 분석 중...")

    system_prompt = (
        "너는 취업 코치 멀티 에이전트 시스템의 supervisor다. "
        "사용자 요청을 분석해서 다음 세 에이전트 중 가장 적합한 하나를 선택하라.\n\n"
        "- news: 채용 시장, 산업 동향, 특정 기업/직무 관련 최신 뉴스가 필요한 경우\n"
        "- rag: 채용공고 상세 내용, 회사 문화, 기술 스택, 면접 FAQ 등 보유 문서 기반 정보가 필요한 경우\n"
        "- resume: 자기소개서/이력서 작성이나 첨삭, 피드백이 필요한 경우\n\n"
        "반드시 news, rag, resume 중 하나의 단어만 소문자로 출력하라. 다른 말은 절대 덧붙이지 마라."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])

    choice = response.content.strip().lower()
    if choice not in AGENT_LABELS:
        choice = "rag"

    print(f"[supervisor_node] 선택된 에이전트 → {AGENT_LABELS[choice]}")
    return {"next_agent": choice}


def news_agent_node(state: SupervisorState) -> SupervisorState:
    query = state["user_input"]
    print(f"[news_agent_node] '{query}' 관련 뉴스 검색 중...")

    news_raw = search_job_news.invoke(query)

    messages = [
        SystemMessage(content=(
            "너는 취업 준비생을 위한 산업/채용 뉴스 코치다. "
            "아래 검색된 뉴스만 근거로 사용자 질문과 관련된 핵심 트렌드와 "
            "취업 준비에 참고할 시사점을 한국어로 정리하라. "
            "뉴스에 없는 내용은 지어내지 마라."
        )),
        HumanMessage(content=f"사용자 질문: {query}\n\n검색된 뉴스:\n{news_raw}"),
    ]
    response = llm.invoke(messages)

    return {"news_result": response.content}


def rag_agent_node(state: SupervisorState) -> SupervisorState:
    query = state["user_input"]
    print(f"[rag_agent_node] 채용 문서 검색 중... (top-{TOP_K})")

    docs = job_retriever.invoke(query)
    if not docs:
        return {"rag_result": "관련 채용 정보를 문서에서 찾지 못했습니다."}

    sources = {d.metadata.get("source_file", "unknown") for d in docs}
    print(f"[rag_agent_node] {len(docs)}개 청크 검색 완료 (출처: {', '.join(sorted(sources))})")

    context = format_context(docs)
    messages = [
        SystemMessage(content=(
            "너는 채용 문서 분석 어시스턴트다. 아래 제공된 발췌 내용만 근거로 한국어로 답변한다.\n"
            "- 발췌에 없는 정보는 지어내지 말고 '문서에서 해당 정보를 찾을 수 없습니다'라고 답한다.\n"
            "- 자격 요건, 근무 조건 등은 원문 그대로 정확히 인용한다.\n"
            "- 답변 끝에 참고한 출처 파일명을 표기한다."
        )),
        HumanMessage(content=f"[문서 발췌]\n{context}\n\n[질문]\n{query}"),
    ]
    response = llm.invoke(messages)

    return {"rag_result": response.content}


def resume_agent_node(state: SupervisorState) -> SupervisorState:
    user_input = state["user_input"]
    print("[resume_agent_node] 자소서/이력서 피드백 생성 중...")

    messages = [
        SystemMessage(content=(
            "너는 채용 담당자 관점의 자소서/이력서 첨삭 전문가다. "
            "사용자가 제공한 자소서 내용이나 관련 질문을 분석해 "
            "강점, 개선점, 문장 단위 수정 제안을 한국어로 제공하라.\n"
            "사용자가 언급하지 않은 경력이나 수치를 새로 만들어내지 마라."
        )),
        HumanMessage(content=user_input),
    ]
    response = llm.invoke(messages)

    return {"resume_result": response.content}


def final_node(state: SupervisorState) -> SupervisorState:
    print("[final_node] 최종 답변 정리 중...")

    next_agent = state["next_agent"]
    result_map = {
        "news": state.get("news_result", ""),
        "rag": state.get("rag_result", ""),
        "resume": state.get("resume_result", ""),
    }
    body = result_map.get(next_agent, "").strip() or "결과를 생성하지 못했습니다."

    final_answer = f"[담당 에이전트: {AGENT_LABELS.get(next_agent, next_agent)}]\n\n{body}"
    return {"final_answer": final_answer}


# ---------------------------------------------------------
# 6. 조건 분기
# ---------------------------------------------------------
def route_agent(state: SupervisorState) -> str:
    return state["next_agent"]


# ---------------------------------------------------------
# 7. 그래프 구성
# ---------------------------------------------------------
def build_graph():
    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("resume_agent", resume_agent_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "news": "news_agent",
            "rag": "rag_agent",
            "resume": "resume_agent",
        },
    )
    workflow.add_edge("news_agent", "final")
    workflow.add_edge("rag_agent", "final")
    workflow.add_edge("resume_agent", "final")
    workflow.add_edge("final", END)

    return workflow.compile()


# ---------------------------------------------------------
# 8. 결과 저장 (채용공고 기반 취업준비 전략 / 자소서 피드백만 대상)
# ---------------------------------------------------------
COACHING_AGENTS = {"rag", "resume"}  # news(단순 뉴스 검색)는 저장 대상에서 제외


def save_result(state: SupervisorState) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"job_coaching_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"질문: {state['user_input']}\n")
        f.write(f"담당 에이전트: {AGENT_LABELS.get(state['next_agent'], state['next_agent'])}\n")
        f.write("=" * 60 + "\n")
        f.write(state["final_answer"] + "\n")

    return filename


def load_resume_text() -> str:
    """docs/자소서.txt 파일을 읽어 자소서 원문을 반환한다."""
    if not os.path.exists(RESUME_FILE):
        return ""
    try:
        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(RESUME_FILE, "r", encoding="cp949") as f:
            return f.read().strip()


# ---------------------------------------------------------
# 9. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()
    EXIT_WORDS = {"종료", "exit", "quit", "q"}
    RESUME_TRIGGERS = {"자소서", "첨삭", "resume"}

    print("=" * 60)
    print("취업 코치 멀티 에이전트")
    print("뉴스 검색 / 채용공고 분석(RAG) / 자소서 피드백을 지원합니다.")
    print(f"자소서 피드백을 받으려면 '{RESUME_FILE}' 파일을 넣어두고 '자소서'를 입력하세요.")
    print("종료하려면 '종료'를 입력하세요.")
    print("=" * 60)

    while True:
        user_input = input("\n질문을 입력하세요: ").strip()

        if user_input in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            print("질문을 입력해 주세요.")
            continue

        # 자소서 피드백은 docs/자소서.txt 파일 내용을 읽어서 사용한다
        if user_input in RESUME_TRIGGERS:
            resume_text = load_resume_text()
            if not resume_text:
                print(f"'{RESUME_FILE}' 파일을 찾을 수 없거나 내용이 비어 있습니다. "
                      f"docs 폴더에 '자소서.txt'를 넣어주세요.")
                continue
            print(f"[안내] '{RESUME_FILE}' 파일을 읽어 자소서 피드백을 요청합니다.")
            user_input = resume_text

        initial_state: SupervisorState = {
            "user_input": user_input,
            "next_agent": "",
            "news_result": "",
            "rag_result": "",
            "resume_result": "",
            "final_answer": "",
        }

        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        print("\n" + "-" * 60)
        print(result["final_answer"])
        print("-" * 60)

        # 취업준비 전략(rag) / 자소서 피드백(resume) 결과만 저장 대상으로 안내
        if result["next_agent"] in COACHING_AGENTS:
            answer = input("\n이 취업 코칭 결과를 txt 파일로 저장하시겠습니까? (y/n): ").strip().lower()
            if answer in {"y", "yes"}:
                saved_path = save_result(result)
                print(f"[저장 완료] 결과가 '{saved_path}' 파일에 저장되었습니다.")
            else:
                print("저장하지 않았습니다.")
