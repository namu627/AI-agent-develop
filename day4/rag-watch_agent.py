import os
import glob
import json
import base64
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
# 1. 환경 설정
# ---------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")

DOCS_DIR = "docs"
IMAGES_DIR = "images"
CAPTION_CACHE = "image_captions.json"
TOP_K = 5

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


# ---------------------------------------------------------
# 2. 이미지 유틸 (A: 캡션 인덱싱 / C: 사진 질의)
# ---------------------------------------------------------
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def media_type_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")


CAPTION_PROMPT = (
    "이 시계 사진을 검색용 설명문으로 변환하라. 한국어로 작성하며 다음을 포함한다.\n"
    "- 추정 브랜드와 모델명 (확신 없으면 '추정' 표기)\n"
    "- 케이스 소재와 형태, 대략적 크기 인상\n"
    "- 다이얼 색상과 인덱스 형태, 베젤 유무와 종류\n"
    "- 브레이슬릿/스트랩 소재\n"
    "- 서브다이얼, 날짜창, GMT 핸즈 등 기능 요소\n"
    "- 전체 인상 (드레스/다이버/파일럿/필드 등 카테고리)\n\n"
    "제약: 보이지 않는 정보(가격, 무브먼트, 생산연도)는 추측하지 마라. "
    "8~12문장의 평문으로 쓰고 목록 기호는 쓰지 마라."
)


def caption_image(path: str) -> str:
    """이미지 → 검색용 설명문 (vision)"""
    b64 = encode_image(path)
    mt = media_type_of(path)

    response = vision_llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
        ])
    ])
    return response.content


def load_image_documents() -> List[Document]:
    """images/ 폴더의 사진을 캡션으로 변환해 Document로 반환. 캐시 사용."""
    if not os.path.isdir(IMAGES_DIR):
        print(f"[setup] '{IMAGES_DIR}' 폴더가 없어 이미지 인덱싱을 건너뜁니다.")
        return []

    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    paths = sorted(paths)

    if not paths:
        print("[setup] 인덱싱할 이미지가 없습니다.")
        return []

    cache = {}
    if os.path.exists(CAPTION_CACHE):
        with open(CAPTION_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    docs, updated = [], False
    for path in paths:
        filename = os.path.basename(path)

        if filename in cache:
            caption = cache[filename]
            print(f"  - {filename} (캐시)")
        else:
            print(f"  - {filename} 분석 중...")
            try:
                caption = caption_image(path)
                cache[filename] = caption
                updated = True
            except Exception as e:
                print(f"    실패: {e}")
                continue

        docs.append(Document(
            page_content=f"[시계 사진 설명 - {filename}]\n{caption}",
            metadata={"source_file": f"images/{filename}", "type": "image"},
        ))

    if updated:
        with open(CAPTION_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    return docs


# ---------------------------------------------------------
# 3. 문서 로드 → 분할 → 임베딩 → FAISS
# ---------------------------------------------------------
def load_text_documents() -> List[Document]:
    if not os.path.isdir(DOCS_DIR):
        raise FileNotFoundError(f"'{DOCS_DIR}' 폴더를 찾을 수 없습니다.")

    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"'{DOCS_DIR}' 폴더에 .txt 파일이 없습니다.")

    all_docs = []
    for path in paths:
        try:
            docs = TextLoader(path, encoding="utf-8").load()
        except UnicodeDecodeError:
            docs = TextLoader(path, encoding="cp949").load()

        filename = os.path.basename(path)
        for d in docs:
            d.metadata["source_file"] = filename
            d.metadata["type"] = "text"
        all_docs.extend(docs)
        print(f"  - {filename} 로드 ({len(docs[0].page_content):,}자)")

    return all_docs


def build_vectorstore() -> FAISS:
    print(f"[setup] 텍스트 문서 로드")
    documents = load_text_documents()

    print(f"[setup] 이미지 캡션 생성")
    documents += load_image_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n■", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[setup] 청크 분할 완료: {len(chunks)}개")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OPENAI_API_KEY
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[setup] FAISS 벡터 저장소 생성 완료\n")
    return vectorstore


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ---------------------------------------------------------
# 4. State 정의
# ---------------------------------------------------------
class RAGState(TypedDict):
    mode: str                       # "qa" | "recommend" | "image"
    question: str                   # 질문 또는 검색 쿼리
    user_profile: str               # 예산·취향 정보 (recommend)
    image_path: str                 # 업로드 이미지 경로 (image)
    image_desc: str                 # vision 식별 결과 (image)
    retrieved_docs: List[Document]
    answer: str


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", "unknown")
        parts.append(f"[발췌 {i} | 출처: {src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------
# 5. 노드 정의
# ---------------------------------------------------------
def vision_node(RAGState_in: RAGState) -> RAGState:
    """C: 사용자가 올린 사진을 식별해 검색 쿼리로 변환한다."""
    state = RAGState_in
    path = state["image_path"]
    print("[vision_node] 사진 분석 중...")

    b64 = encode_image(path)
    mt = media_type_of(path)

    prompt = (
        "이 시계 사진을 보고 다음을 한국어로 답하라.\n"
        "1) 추정 브랜드와 모델 (확신도를 상/중/하로 표기)\n"
        "2) 근거가 된 시각적 특징 3가지\n"
        "3) 카테고리 (다이버/드레스/파일럿/필드/크로노그래프 등)\n"
        "4) 다이얼 색상, 케이스 소재, 스트랩 종류\n\n"
        "사진에서 확인 불가능한 정보는 '확인 불가'로 표기하고 추측하지 마라."
    )

    response = vision_llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
        ])
    ])

    desc = response.content
    print("[vision_node] 식별 완료")

    return {
        "image_desc": desc,
        "question": f"{desc}\n\n이 시계의 스펙, 가격대, 특징",
    }


def retrieve_node(state: RAGState) -> RAGState:
    query = state["question"]
    print(f"[retrieve_node] 관련 문서 검색 중... (top-{TOP_K})")

    scored = vectorstore.similarity_search_with_score(query, k=TOP_K)
    for i, (doc, score) in enumerate(scored, 1):
        src = doc.metadata.get("source_file", "unknown")
        preview = doc.page_content[:50].replace("\n", " ")
        print(f"    {i}. [{score:.3f}] ({src}) {preview}...")

    return {"retrieved_docs": [d for d, _ in scored]}


def generate_node(state: RAGState) -> RAGState:
    docs = state["retrieved_docs"]
    print("[generate_node] 답변 생성 중...")

    if not docs:
        return {"answer": "관련 정보를 문서에서 찾지 못했습니다."}

    system_prompt = (
        "너는 명품 시계 상담 어시스턴트다. 아래 발췌 내용만 근거로 한국어로 답한다.\n"
        "- 발췌에 없는 정보는 지어내지 말고 '자료에서 확인되지 않습니다'라고 답한다.\n"
        "- 가격, 사이즈, 방수 등급 같은 수치는 원문 그대로 인용한다.\n"
        "- 가격은 2026년 상반기 참고치이며 변동 가능하다는 점을 함께 알린다.\n"
        "- 답변 끝에 참고 출처 파일명을 표기한다."
    )
    user_prompt = f"[자료 발췌]\n{format_context(docs)}\n\n[질문]\n{state['question']}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"answer": response.content}


def recommend_node(state: RAGState) -> RAGState:
    """예산·취향 기반 모델 추천"""
    docs = state["retrieved_docs"]
    profile = state["user_profile"]
    print("[recommend_node] 추천 생성 중...")

    if not docs:
        return {"answer": "추천에 필요한 자료를 찾지 못했습니다."}

    system_prompt = (
        "너는 명품 시계 구매 상담 전문가다. 제공된 자료 발췌만 근거로 "
        "고객 조건에 맞는 모델을 추천한다.\n\n"
        "다음 형식으로 한국어로 답하라.\n\n"
        "## 1. 조건 요약\n"
        "고객이 밝힌 예산, 용도, 취향, 손목 둘레를 한 줄로 정리한다.\n\n"
        "## 2. 추천 모델 3종\n"
        "각 모델마다 아래를 표기한다.\n"
        "- 모델명과 가격 (자료에 나온 수치 그대로)\n"
        "- 케이스 크기, 무브먼트, 방수 등급\n"
        "- 이 고객에게 맞는 이유 (조건과 직접 연결해 설명)\n"
        "- 감안할 점 (단점이나 구매 난이도)\n"
        "추천 순서는 조건 적합도가 높은 순으로 한다.\n\n"
        "## 3. 최종 한 줄 추천\n"
        "셋 중 하나를 고르고 이유를 두 문장으로 요약한다.\n\n"
        "## 4. 구매 전 확인 사항\n"
        "자료에 있는 사이즈 선택 기준, 구매 채널, 유지비 관련 조언 중 "
        "이 고객에게 해당하는 것을 2~3가지 제시한다.\n\n"
        "[엄격한 제약]\n"
        "- 자료 발췌에 등장하지 않는 모델은 추천하지 않는다.\n"
        "- 가격을 임의로 만들지 않는다. 자료에 가격이 없으면 '자료에 가격 정보 없음'이라 쓴다.\n"
        "- 예산 범위에 맞는 모델이 자료에 부족하면, 억지로 3개를 채우지 말고 "
        "가능한 개수만 제시한 뒤 그 사실을 밝힌다.\n"
        "- 시계를 투자 수단으로 권유하지 않는다."
    )
    user_prompt = f"[자료 발췌]\n{format_context(docs)}\n\n[고객 조건]\n{profile}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"answer": response.content}


def image_answer_node(state: RAGState) -> RAGState:
    """사진 식별 결과 + 검색 자료를 합쳐 답변"""
    docs = state["retrieved_docs"]
    print("[image_answer_node] 답변 생성 중...")

    system_prompt = (
        "너는 명품 시계 감별 상담 어시스턴트다. "
        "사진 분석 결과와 자료 발췌를 종합해 한국어로 답한다.\n\n"
        "## 1. 사진 분석 결과\n"
        "추정 모델과 확신도, 근거가 된 특징을 정리한다.\n\n"
        "## 2. 해당 모델 정보\n"
        "자료에서 확인되는 스펙과 가격을 제시한다. "
        "자료에 없는 모델이면 '보유 자료에 없는 모델입니다'라고 밝히고, "
        "사진에서 파악된 특징만 설명한다.\n\n"
        "## 3. 비슷한 대안\n"
        "자료에 있는 모델 중 유사한 성격의 것을 1~2개 소개한다.\n\n"
        "[제약]\n"
        "- 사진만으로 정품 여부는 판정할 수 없다. 요청받아도 단정하지 말고 "
        "자료에 있는 중고 확인 사항을 안내하는 선에서 그친다.\n"
        "- 자료에 없는 가격이나 스펙을 만들지 않는다."
    )
    user_prompt = (
        f"[사진 분석 결과]\n{state['image_desc']}\n\n"
        f"[자료 발췌]\n{format_context(docs)}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"answer": response.content}


# ---------------------------------------------------------
# 6. 조건 분기
# ---------------------------------------------------------
def route_entry(state: RAGState) -> str:
    return "vision" if state["mode"] == "image" else "retrieve"


def route_after_retrieve(state: RAGState) -> str:
    mode = state["mode"]
    if mode == "recommend":
        return "recommend"
    if mode == "image":
        return "image_answer"
    return "generate"


# ---------------------------------------------------------
# 7. 그래프 구성
# ---------------------------------------------------------
def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("vision", vision_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("recommend", recommend_node)
    workflow.add_node("image_answer", image_answer_node)

    workflow.add_conditional_edges(
        START, route_entry, {"vision": "vision", "retrieve": "retrieve"}
    )
    workflow.add_edge("vision", "retrieve")
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {
            "generate": "generate",
            "recommend": "recommend",
            "image_answer": "image_answer",
        },
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("recommend", END)
    workflow.add_edge("image_answer", END)

    return workflow.compile()


# ---------------------------------------------------------
# 8. 입력 헬퍼
# ---------------------------------------------------------
def ask(label: str, default: str = "") -> str:
    val = input(f"  {label}: ").strip()
    return val if val else default


def collect_profile() -> str:
    print("\n[구매 상담] 아래 항목을 입력하세요. 모르면 엔터로 건너뛰면 됩니다.")
    budget = ask("예산 (예: 500만 원, 300~700만)", "미지정")
    purpose = ask("주 용도 (오피스/데일리/스포츠/행사)", "미지정")
    style = ask("선호 스타일 (다이버/드레스/파일럿/필드/무관)", "무관")
    wrist = ask("손목 둘레 cm (모르면 엔터)", "미지정")
    brand = ask("관심 브랜드 (없으면 엔터)", "무관")
    etc = ask("기타 조건 (첫 시계 여부, 중고 가능 여부 등)", "없음")

    return (
        f"예산: {budget}\n주 용도: {purpose}\n선호 스타일: {style}\n"
        f"손목 둘레: {wrist}\n관심 브랜드: {brand}\n기타: {etc}"
    )


# ---------------------------------------------------------
# 9. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()
    EXIT_WORDS = {"종료", "exit", "quit", "q"}

    print("=" * 62)
    print("명품 시계 RAG 상담 에이전트")
    print("-" * 62)
    print("  1) 그냥 질문 입력   → 자료 기반 답변")
    print("  2) '추천' 입력      → 예산·취향 기반 모델 추천")
    print("  3) '사진' 입력      → 시계 사진으로 모델 식별")
    print("  4) '종료' 입력      → 프로그램 종료")
    print("=" * 62)

    while True:
        user_input = input("\n입력: ").strip()

        if user_input in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            continue

        # ---- 추천 모드 ----
        if user_input in {"추천", "상담", "recommend"}:
            profile = collect_profile()
            state: RAGState = {
                "mode": "recommend",
                "question": f"예산과 용도에 맞는 시계 모델 추천\n{profile}",
                "user_profile": profile,
                "image_path": "",
                "image_desc": "",
                "retrieved_docs": [],
                "answer": "",
            }

        # ---- 사진 모드 ----
        elif user_input in {"사진", "이미지", "image"}:
            path = input("  이미지 경로 (예: images/watch1.jpg): ").strip().strip('"')
            if not os.path.exists(path):
                print(f"  파일을 찾을 수 없습니다: {path}")
                continue
            state = {
                "mode": "image",
                "question": "",
                "user_profile": "",
                "image_path": path,
                "image_desc": "",
                "retrieved_docs": [],
                "answer": "",
            }

        # ---- 일반 질문 ----
        else:
            state = {
                "mode": "qa",
                "question": user_input,
                "user_profile": "",
                "image_path": "",
                "image_desc": "",
                "retrieved_docs": [],
                "answer": "",
            }

        print("-" * 62)
        try:
            result = graph.invoke(state)
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

        print("-" * 62)
        print(result["answer"])
        print("=" * 62)