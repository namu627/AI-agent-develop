import os
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

DOC_PATH = "job_posting.txt"
TOP_K = 3


# ---------------------------------------------------------
# 2. 문서 로드 → 분할 → 임베딩 → FAISS 저장
# ---------------------------------------------------------
def build_vectorstore() -> FAISS:
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError(f"'{DOC_PATH}' 파일을 찾을 수 없습니다.")

    # 2-1. 문서 로드
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()
    print(f"[setup] 문서 로드 완료: {len(documents)}개")

    # 2-2. 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[setup] 청크 분할 완료: {len(chunks)}개")

    # 2-3. 임베딩 + FAISS 인덱싱
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
    question: str                  # 사용자 질문
    retrieved_docs: List[Document] # 검색된 문서 청크
    answer: str                    # 최종 답변


# ---------------------------------------------------------
# 4. 노드 정의
# ---------------------------------------------------------
def retrieve_node(state: RAGState) -> RAGState:
    """벡터 DB에서 질문과 관련된 문서를 검색한다."""
    question = state["question"]
    print(f"[retrieve_node] 관련 문서 검색 중... (top-{TOP_K})")

    docs = retriever.invoke(question)
    print(f"[retrieve_node] {len(docs)}개 청크 검색 완료")

    return {"retrieved_docs": docs}


def generate_node(state: RAGState) -> RAGState:
    """검색된 문서를 근거로 LLM이 답변을 생성한다."""
    question = state["question"]
    docs = state["retrieved_docs"]
    print("[generate_node] 답변 생성 중...")

    if not docs:
        return {"answer": "관련 정보를 채용공고에서 찾지 못했습니다."}

    context = "\n\n---\n\n".join(
        f"[청크 {i}]\n{d.page_content}" for i, d in enumerate(docs, 1)
    )

    system_prompt = (
        "너는 채용공고 안내 어시스턴트다. 아래 제공된 채용공고 발췌 내용만 근거로 "
        "한국어로 답변한다.\n"
        "- 발췌 내용에 없는 정보는 추측하거나 지어내지 말고, "
        "'채용공고에서 해당 정보를 찾을 수 없습니다'라고 답한다.\n"
        "- 급여, 날짜, 인원 같은 수치는 원문 그대로 정확히 인용한다.\n"
        "- 답변은 간결하게 정리하되, 항목이 여러 개면 목록으로 보여준다."
    )

    user_prompt = f"[채용공고 발췌]\n{context}\n\n[질문]\n{question}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {"answer": response.content}


# ---------------------------------------------------------
# 5. 그래프 구성
# ---------------------------------------------------------
def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ---------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    EXIT_WORDS = {"종료", "exit", "quit", "q"}

    print("=" * 60)
    print("채용공고 RAG 에이전트")
    print("예) AI 엔지니어 자격 요건이 뭐야? / 재택근무 되나요? / 접수 마감일 언제야?")
    print("종료하려면 '종료' 를 입력하세요.")
    print("=" * 60)

    while True:
        question = input("\n질문을 입력하세요: ").strip()

        if question in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break
        if not question:
            print("질문을 입력해 주세요.")
            continue

        initial_state: RAGState = {
            "question": question,
            "retrieved_docs": [],
            "answer": "",
        }

        print("-" * 60)
        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        print("-" * 60)
        print("[답변]")
        print(result["answer"])
        print("=" * 60)