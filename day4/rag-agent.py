import os
import glob
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------
# 1. 환경변수 로드
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")

DOCS_DIR = "docs"       # txt 파일들이 들어있는 폴더
TOP_K = 4


# ---------------------------------------------------------
# 2. 여러 txt 로드 → 분할 → 임베딩 → FAISS
# ---------------------------------------------------------
def load_all_documents(docs_dir: str) -> List[Document]:
    """폴더 내 모든 .txt 파일을 로드하고 출처 메타데이터를 붙인다."""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"'{docs_dir}' 폴더를 찾을 수 없습니다.")

    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"'{docs_dir}' 폴더에 .txt 파일이 없습니다.")

    all_docs: List[Document] = []
    for path in paths:
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
        except UnicodeDecodeError:
            # Windows 메모장 등에서 저장한 cp949 파일 대응
            loader = TextLoader(path, encoding="cp949")
            docs = loader.load()

        filename = os.path.basename(path)
        for d in docs:
            d.metadata["source_file"] = filename

        all_docs.extend(docs)
        print(f"  - {filename} 로드 ({len(docs[0].page_content):,}자)")

    return all_docs


def build_vectorstore() -> FAISS:
    print(f"[setup] '{DOCS_DIR}' 폴더에서 문서 로드 중...")
    documents = load_all_documents(DOCS_DIR)
    print(f"[setup] 총 {len(documents)}개 문서 로드 완료")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[setup] 청크 분할 완료: {len(chunks)}개")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[setup] FAISS 벡터 저장소 생성 완료\n")

    return vectorstore


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ---------------------------------------------------------
# 3. State 정의
# ---------------------------------------------------------
class RAGState(TypedDict):
    mode: str                       # "qa" | "review"
    question: str                   # 질문 또는 검색 쿼리
    cover_letter: str               # 자소서 원문 (review 모드)
    retrieved_docs: List[Document]  # 검색된 청크
    answer: str                     # 최종 출력


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", "unknown")
        parts.append(f"[발췌 {i} | 출처: {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------
# 4. 노드 정의
# ---------------------------------------------------------
def retrieve_node(state: RAGState) -> RAGState:
    """벡터 DB에서 관련 문서를 검색한다."""
    query = state["question"]
    print(f"[retrieve_node] 관련 문서 검색 중... (top-{TOP_K})")

    docs = retriever.invoke(query)

    sources = {d.metadata.get("source_file", "unknown") for d in docs}
    print(f"[retrieve_node] {len(docs)}개 청크 검색 완료 (출처: {', '.join(sorted(sources))})")

    return {"retrieved_docs": docs}


def generate_node(state: RAGState) -> RAGState:
    """검색 결과를 근거로 답변을 생성한다."""
    docs = state["retrieved_docs"]
    print("[generate_node] 답변 생성 중...")

    if not docs:
        return {"answer": "관련 정보를 문서에서 찾지 못했습니다."}

    context = format_context(docs)

    system_prompt = (
        "너는 채용 문서 안내 어시스턴트다. 아래 제공된 발췌 내용만 근거로 한국어로 답변한다.\n"
        "- 발췌에 없는 정보는 지어내지 말고 '문서에서 해당 정보를 찾을 수 없습니다'라고 답한다.\n"
        "- 급여, 날짜, 인원 같은 수치는 원문 그대로 정확히 인용한다.\n"
        "- 답변 끝에 참고한 출처 파일명을 표기한다."
    )
    user_prompt = f"[문서 발췌]\n{context}\n\n[질문]\n{state['question']}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"answer": response.content}


def review_node(state: RAGState) -> RAGState:
    """검색된 채용 정보를 기준으로 자소서를 첨삭한다."""
    docs = state["retrieved_docs"]
    cover_letter = state["cover_letter"]
    print("[review_node] 자소서 분석 중...")

    if not docs:
        return {"answer": "채용 정보를 찾지 못해 첨삭할 수 없습니다."}

    context = format_context(docs)

    system_prompt = (
        "너는 채용 담당자 관점의 자소서 첨삭 전문가다. "
        "제공된 채용공고 발췌를 유일한 기준으로 삼아 지원자의 자소서를 평가하고 개선한다.\n\n"
        "아래 형식으로 한국어로 답하라.\n\n"
        "## 1. 총평\n"
        "공고 적합도를 100점 만점으로 매기고 2~3문장으로 근거를 설명한다.\n\n"
        "## 2. 공고와 잘 맞는 부분\n"
        "자소서의 어떤 문장이 공고의 어떤 요건과 연결되는지 짝지어 제시한다.\n\n"
        "## 3. 부족하거나 빠진 부분\n"
        "공고의 필수/우대 요건 중 자소서에서 다루지 않은 항목을 지적한다. "
        "지원자가 실제로 그 경험이 있는지는 알 수 없으므로, "
        "'경험이 있다면 이렇게 드러내라'는 식으로 제안한다.\n\n"
        "## 4. 문장 단위 개선 제안\n"
        "원문 → 수정안 형태로 3~5개 제시한다. 추상적 표현을 구체적 수치와 행동으로 바꾸는 데 집중한다.\n\n"
        "## 5. 개선된 자소서 초안\n"
        "위 내용을 반영한 전체 수정본을 작성한다.\n\n"
        "[엄격한 제약]\n"
        "- 지원자가 쓰지 않은 경력, 수치, 프로젝트를 새로 만들어내지 않는다. "
        "구체화가 필요한 자리는 [예: 응답속도 30% 개선]처럼 대괄호 placeholder로 표시한다.\n"
        "- 공고 발췌에 없는 회사 정보를 추측해 언급하지 않는다."
    )

    user_prompt = (
        f"[채용공고 발췌]\n{context}\n\n"
        f"[지원자 자소서 원문]\n{cover_letter}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"answer": response.content}


# ---------------------------------------------------------
# 5. 조건 분기
# ---------------------------------------------------------
def route_mode(state: RAGState) -> str:
    return "review" if state["mode"] == "review" else "generate"


# ---------------------------------------------------------
# 6. 그래프 구성
# ---------------------------------------------------------
def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_mode,
        {"generate": "generate", "review": "review"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("review", END)

    return workflow.compile()


# ---------------------------------------------------------
# 7. 입력 헬퍼
# ---------------------------------------------------------
def read_multiline(prompt: str) -> str:
    """여러 줄 입력을 받는다. 빈 줄에서 'END' 입력 시 종료."""
    print(prompt)
    print("(입력이 끝나면 새 줄에 END 를 입력하세요)")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ---------------------------------------------------------
# 8. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    EXIT_WORDS = {"종료", "exit", "quit", "q"}

    print("=" * 60)
    print("채용 문서 RAG 에이전트")
    print("-" * 60)
    print("  1) 그냥 질문 입력  → 채용 문서 기반 답변")
    print("  2) '자소서' 입력   → 자소서 첨삭 모드")
    print("  3) '종료' 입력     → 프로그램 종료")
    print("=" * 60)

    while True:
        user_input = input("\n입력: ").strip()

        if user_input in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            continue

        # ---- 자소서 첨삭 모드 ----
        if user_input in {"자소서", "첨삭", "review"}:
            target = input("지원 직무를 입력하세요 (예: AI 엔지니어): ").strip()
            if not target:
                print("직무를 입력해야 합니다.")
                continue

            cover_letter = read_multiline("\n자소서를 붙여넣으세요:")
            if not cover_letter:
                print("자소서 내용이 비어 있습니다.")
                continue

            state: RAGState = {
                "mode": "review",
                "question": f"{target} 모집 부문의 담당 업무, 필수 자격, 우대 사항",
                "cover_letter": cover_letter,
                "retrieved_docs": [],
                "answer": "",
            }
        # ---- 일반 질문 모드 ----
        else:
            state = {
                "mode": "qa",
                "question": user_input,
                "cover_letter": "",
                "retrieved_docs": [],
                "answer": "",
            }

        print("-" * 60)
        try:
            result = graph.invoke(state)
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        print("-" * 60)
        print(result["answer"])
        print("=" * 60)