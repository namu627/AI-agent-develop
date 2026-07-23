import os
import re
import sys
import json
import glob
import base64
from datetime import datetime
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
    print("[오류] .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다. 프로그램을 종료합니다.")
    sys.exit(1)

DOCS_DIR = "docs"
IMAGES_DIR = "images"
CAPTION_CACHE = "image_captions.json"
OUTPUTS_DIR = "outputs"
TOP_K = 5
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

# docs 폴더에는 다른 실습(취업 코치)용 문서가 함께 들어 있어, 시계 상담에 쓸
# 4개 파일만 명시적으로 지정해 로드한다. (glob으로 전체를 읽으면 무관한 문서가 섞여
# RAG 검색 결과가 오염된다)
WATCH_DOC_FILES = ["watch_brands.txt", "watch_models.txt", "watch_knowledge.txt", "watch_market.txt"]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


# ---------------------------------------------------------
# 2. 이미지 유틸 (캡션 인덱싱 / 사진 질의 공용)
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

    all_docs: List[Document] = []
    for filename in WATCH_DOC_FILES:
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(path):
            print(f"[setup] 경고: '{path}' 파일을 찾을 수 없어 건너뜁니다.")
            continue

        try:
            docs = TextLoader(path, encoding="utf-8").load()
        except UnicodeDecodeError:
            docs = TextLoader(path, encoding="cp949").load()

        for d in docs:
            d.metadata["source_file"] = filename
            d.metadata["type"] = "text"
        all_docs.extend(docs)
        print(f"  - {filename} 로드 ({len(docs[0].page_content):,}자)")

    if not all_docs:
        raise FileNotFoundError(f"'{DOCS_DIR}' 폴더에서 시계 관련 문서를 찾지 못했습니다.")

    return all_docs


def build_vectorstore() -> FAISS:
    print("[setup] 텍스트 문서 로드")
    documents = load_text_documents()

    print("[setup] 이미지 캡션 생성")
    documents += load_image_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n■", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[setup] 청크 분할 완료: {len(chunks)}개")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[setup] FAISS 벡터 저장소 생성 완료\n")
    return vectorstore


vectorstore = build_vectorstore()


def retrieve(query: str, k: int = TOP_K) -> List[Document]:
    """벡터 DB에서 관련 문서를 점수와 함께 검색하고, 문서 목록만 반환한다."""
    print(f"[retrieve] 관련 문서 검색 중... (top-{k})")
    scored = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(scored, 1):
        src = doc.metadata.get("source_file", "unknown")
        preview = doc.page_content[:50].replace("\n", " ")
        print(f"    {i}. [{score:.3f}] ({src}) {preview}...")
    return [d for d, _ in scored]


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
    routing_reason: str
    market_result: str
    recommend_result: str
    image_result: str
    care_result: str
    image_path: str
    retrieved_docs: List[Document]
    final_answer: str


VALID_AGENTS = {"market", "recommend", "image", "care"}

AGENT_LABELS = {
    "market": "📊 브랜드·시장 정보 에이전트",
    "recommend": "🎯 구매 추천 에이전트",
    "image": "📷 이미지 감정 에이전트",
    "care": "🛠️ 관리·유지보수 에이전트",
}

PRICE_PATTERN = re.compile(r"\d[\d,]*\s?(만\s?원|원|\$)")


# ---------------------------------------------------------
# 5. 노드 정의
# ---------------------------------------------------------
def parse_supervisor_response(raw: str) -> tuple[str, str]:
    """LLM의 JSON 응답을 파싱한다. 실패 시 (market, 기본 사유)로 폴백."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    agent, reason = "", ""
    try:
        data = json.loads(cleaned)
        agent = str(data.get("agent", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
    except Exception:
        pass

    if agent not in VALID_AGENTS:
        agent = "market"
        reason = reason or "응답 파싱 실패로 기본 에이전트(market)로 처리합니다."

    return agent, reason


def supervisor_node(state: SupervisorState) -> SupervisorState:
    user_input = state["user_input"]
    print("[supervisor] 요청 분석 중...")

    system_prompt = (
        "너는 명품 시계 상담 멀티 에이전트 시스템의 supervisor다. "
        "사용자 요청을 분석해서 다음 네 에이전트 중 가장 적합한 하나를 선택하라.\n\n"
        "- market: 브랜드/모델 정보, 스펙, 가격, 시장 동향 질문\n"
        "- recommend: 예산·용도·취향 기반 모델 추천 요청\n"
        "- image: 시계 사진 식별 및 분석 요청\n"
        "- care: 관리, 오버홀, 방수, 자성, 중고 구매 체크리스트 등 유지보수 질문\n\n"
        "반드시 아래 JSON 형식으로만 응답하라. 다른 텍스트는 절대 포함하지 마라.\n"
        '{"agent": "market|recommend|image|care 중 하나", "reason": "선택 이유 한 문장"}'
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])

    agent, reason = parse_supervisor_response(response.content)
    print(f"[supervisor] '{agent}' 에이전트 선택 → {reason}")

    return {"next_agent": agent, "routing_reason": reason}


def market_agent_node(state: SupervisorState) -> SupervisorState:
    print("[market_agent] 브랜드·시장 정보 검색 중...")
    query = state["user_input"]
    docs = retrieve(query)

    if not docs:
        return {"market_result": "자료에서 확인되지 않습니다.", "retrieved_docs": docs}

    system_prompt = (
        "너는 명품 시계 브랜드·시장 정보 전문가다. 아래 자료 발췌만 근거로 한국어로 답한다.\n"
        "- 발췌에 없는 정보는 지어내지 말고 '자료에서 확인되지 않습니다'라고 답한다.\n"
        "- 가격, 사이즈, 방수 등급 등 수치는 원문 그대로 인용한다.\n"
        "- 가격은 2026년 상반기 참고치이며 변동될 수 있다는 점을 함께 알린다.\n"
        "- 시계를 투자 수단으로 권유하지 않는다.\n"
        "- 정품 감정은 브랜드 A/S 센터에서만 확정 가능하다고 안내한다."
    )
    user_prompt = f"[자료 발췌]\n{format_context(docs)}\n\n[질문]\n{query}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"market_result": response.content, "retrieved_docs": docs}


def recommend_agent_node(state: SupervisorState) -> SupervisorState:
    print("[recommend_agent] 추천 조건 기반 문서 검색 중...")
    query = state["user_input"]
    docs = retrieve(query)

    if not docs:
        return {"recommend_result": "추천에 필요한 자료를 찾지 못했습니다.", "retrieved_docs": docs}

    system_prompt = (
        "너는 명품 시계 구매 상담 전문가다. 제공된 자료 발췌만 근거로 "
        "고객 조건에 맞는 모델을 추천한다.\n\n"
        "다음 형식으로 한국어로 답하라.\n\n"
        "## 조건 요약\n"
        "고객이 밝힌 예산, 용도, 취향, 손목 둘레를 한 줄로 정리한다. "
        "언급되지 않은 항목은 '미언급'으로 표기한다.\n\n"
        "## 추천 모델 3종\n"
        "각 모델마다 아래를 표기한다.\n"
        "- 모델명과 가격 (자료에 나온 수치 그대로)\n"
        "- 케이스 크기, 무브먼트, 방수 등급\n"
        "- 이 고객에게 맞는 이유 (조건과 직접 연결해 설명)\n"
        "- 감안할 점 (단점이나 구매 난이도)\n"
        "추천 순서는 조건 적합도가 높은 순으로 한다.\n\n"
        "## 최종 추천\n"
        "셋 중 하나를 고르고 이유를 두 문장으로 요약한다.\n\n"
        "## 구매 전 확인 사항\n"
        "자료에 있는 사이즈 선택 기준, 구매 채널, 유지비 관련 조언 중 "
        "이 고객에게 해당하는 것을 2~3가지 제시한다.\n\n"
        "[엄격한 제약]\n"
        "- 자료 발췌에 등장하지 않는 모델은 추천하지 않는다.\n"
        "- 가격을 임의로 만들지 않는다. 자료에 가격이 없으면 '자료에 가격 정보 없음'이라 쓴다.\n"
        "- 예산에 맞는 모델이 자료에 부족하면 억지로 3개를 채우지 말고 "
        "가능한 개수만 제시한 뒤 그 사실을 밝힌다.\n"
        "- 시계를 투자 수단으로 권유하지 않는다."
    )
    user_prompt = f"[자료 발췌]\n{format_context(docs)}\n\n[고객 요청]\n{query}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"recommend_result": response.content, "retrieved_docs": docs}


def image_agent_node(state: SupervisorState) -> SupervisorState:
    print("[image_agent] 시계 사진 분석 중...")
    image_path = state.get("image_path", "")

    if not image_path:
        return {
            "image_result": "사진 파일 경로가 없습니다. 메인 메뉴에서 '사진' 또는 '이미지'를 입력해 다시 시도해 주세요.",
            "retrieved_docs": [],
        }

    b64 = encode_image(image_path)
    mt = media_type_of(image_path)

    vision_prompt = (
        "이 시계 사진을 보고 다음을 한국어로 답하라.\n"
        "1) 추정 브랜드와 모델 (확신도를 상/중/하로 표기)\n"
        "2) 근거가 된 시각적 특징 3가지\n"
        "3) 카테고리 (다이버/드레스/파일럿/필드/크로노그래프 등)\n"
        "4) 다이얼 색상, 케이스 소재, 스트랩 종류\n\n"
        "사진에서 확인 불가능한 정보는 '확인 불가'로 표기하고 추측하지 마라."
    )
    vision_response = vision_llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
        ])
    ])
    image_desc = vision_response.content
    print("[image_agent] 사진 분석 완료, 관련 문서 검색 중...")

    query = f"{image_desc}\n\n이 시계의 스펙, 가격대, 특징"
    docs = retrieve(query)

    system_prompt = (
        "너는 명품 시계 감별 상담 어시스턴트다. 사진 분석 결과와 자료 발췌를 종합해 "
        "한국어로 답한다.\n\n"
        "## 사진 분석\n"
        "추정 모델과 확신도, 근거가 된 특징을 정리한다.\n\n"
        "## 해당 모델 정보\n"
        "자료에서 확인되는 스펙과 가격을 제시한다. 자료에 없는 모델이면 "
        "'보유 자료에 없는 모델입니다'라고 밝히고 사진에서 파악된 특징만 설명한다.\n\n"
        "## 비슷한 대안\n"
        "자료에 있는 모델 중 유사한 성격의 것을 1~2개 소개한다.\n\n"
        "[제약]\n"
        "- 사진만으로 정품 여부는 판정할 수 없다. 요청받아도 단정하지 말고, "
        "정품 감정은 브랜드 A/S 센터에서만 확정 가능하다고 안내하며 중고 확인 체크리스트를 "
        "안내하는 선에서 그친다.\n"
        "- 자료에 없는 가격이나 스펙을 만들지 않는다.\n"
        "- 시계를 투자 수단으로 권유하지 않는다."
    )
    user_prompt = f"[사진 분석 결과]\n{image_desc}\n\n[자료 발췌]\n{format_context(docs)}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"image_result": response.content, "retrieved_docs": docs}


def care_agent_node(state: SupervisorState) -> SupervisorState:
    print("[care_agent] 관리·유지보수 관련 문서 검색 중...")
    query = state["user_input"]
    docs = retrieve(query)

    if not docs:
        return {"care_result": "자료에서 확인되지 않습니다.", "retrieved_docs": docs}

    system_prompt = (
        "너는 명품 시계 관리·유지보수 상담 전문가다. 아래 자료 발췌만 근거로 "
        "오버홀 주기·비용, 방수 등급의 실제 의미, 자성 주의 사항, 중고 구매 확인 사항 등을 "
        "한국어로 답한다.\n"
        "- 발췌에 없는 정보는 지어내지 말고 '자료에서 확인되지 않습니다'라고 답한다.\n"
        "- 비용, 주기, 등급 등 수치는 원문 그대로 인용한다.\n"
        "- 정품 감정은 브랜드 A/S 센터에서만 확정 가능하다고 안내한다."
    )
    user_prompt = f"[자료 발췌]\n{format_context(docs)}\n\n[질문]\n{query}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return {"care_result": response.content, "retrieved_docs": docs}


def final_node(state: SupervisorState) -> SupervisorState:
    print("[final] 최종 답변 정리 중...")

    next_agent = state["next_agent"]
    result_map = {
        "market": state.get("market_result", ""),
        "recommend": state.get("recommend_result", ""),
        "image": state.get("image_result", ""),
        "care": state.get("care_result", ""),
    }
    body = result_map.get(next_agent, "").strip() or "결과를 생성하지 못했습니다."

    docs = state.get("retrieved_docs") or []
    sources = sorted({d.metadata.get("source_file", "unknown") for d in docs})

    parts = [f"[담당 에이전트: {AGENT_LABELS.get(next_agent, next_agent)}]", "", body]

    if sources:
        parts += ["", f"참고 출처: {', '.join(sources)}"]

    if PRICE_PATTERN.search(body):
        parts += ["", "※ 가격 정보는 2026년 상반기 기준 참고치이며 실제 거래가와 다를 수 있습니다."]

    return {"final_answer": "\n".join(parts)}


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
    workflow.add_node("market_agent", market_agent_node)
    workflow.add_node("recommend_agent", recommend_agent_node)
    workflow.add_node("image_agent", image_agent_node)
    workflow.add_node("care_agent", care_agent_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "market": "market_agent",
            "recommend": "recommend_agent",
            "image": "image_agent",
            "care": "care_agent",
        },
    )
    workflow.add_edge("market_agent", "final")
    workflow.add_edge("recommend_agent", "final")
    workflow.add_edge("image_agent", "final")
    workflow.add_edge("care_agent", "final")
    workflow.add_edge("final", END)

    return workflow.compile()


# ---------------------------------------------------------
# 8. 결과 저장
# ---------------------------------------------------------
def save_result(state: SupervisorState) -> str:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    now = datetime.now()
    filename = os.path.join(OUTPUTS_DIR, f"watch_consult_{now.strftime('%Y%m%d_%H%M%S')}.txt")

    docs = state.get("retrieved_docs") or []
    sources = sorted({d.metadata.get("source_file", "unknown") for d in docs})

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"질문 일시: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"사용자 입력: {state['user_input']}\n")
        f.write(f"선택된 에이전트: {AGENT_LABELS.get(state['next_agent'], state['next_agent'])}\n")
        f.write(f"선택 이유: {state.get('routing_reason', '')}\n")
        f.write(f"참고 출처: {', '.join(sources) if sources else '없음'}\n")
        f.write("=" * 60 + "\n")
        f.write(state["final_answer"] + "\n")

    return filename


# ---------------------------------------------------------
# 9. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()
    EXIT_WORDS = {"종료", "exit", "quit", "q"}
    IMAGE_TRIGGERS = {"사진", "이미지"}
    SAVE_TRIGGERS = {"저장", "저장해줘", "저장해", "save"}

    print("=" * 62)
    print("명품 시계 상담 멀티 에이전트 (Supervisor 패턴)")
    print("-" * 62)
    print("  - 브랜드/시장/스펙 질문         → market 에이전트")
    print("  - 예산·취향 기반 추천 요청      → recommend 에이전트")
    print("  - '사진' 또는 '이미지' 입력      → image 에이전트 (경로 입력 필요)")
    print("  - 관리/오버홀/방수/중고 체크리스트 → care 에이전트")
    print("  - '저장' 입력                  → 직전 답변을 파일로 저장")
    print("  - '종료' 입력                  → 프로그램 종료")
    print("=" * 62)

    last_result: SupervisorState | None = None

    while True:
        user_input = input("\n입력: ").strip()

        if user_input in EXIT_WORDS:
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            print("입력이 비어 있습니다. 다시 입력해 주세요.")
            continue

        # 사용자가 명시적으로 요청할 때만 직전 답변을 저장한다
        if user_input in SAVE_TRIGGERS:
            if last_result is None:
                print("저장할 이전 답변이 없습니다. 먼저 질문을 입력해 주세요.")
            else:
                saved_path = save_result(last_result)
                print(f"[저장 완료] 상담 결과가 '{saved_path}' 파일에 저장되었습니다.")
            continue

        image_path = ""
        if user_input in IMAGE_TRIGGERS:
            path = input("  이미지 경로를 입력하세요 (예: images/watch1.jpg): ").strip().strip('"')
            if not os.path.exists(path):
                print(f"  파일을 찾을 수 없습니다: {path}")
                continue
            image_path = path
            user_input = "첨부한 시계 사진을 분석하고 관련 정보를 알려줘"

        initial_state: SupervisorState = {
            "user_input": user_input,
            "next_agent": "",
            "routing_reason": "",
            "market_result": "",
            "recommend_result": "",
            "image_result": "",
            "care_result": "",
            "image_path": image_path,
            "retrieved_docs": [],
            "final_answer": "",
        }

        try:
            result = graph.invoke(initial_state)
        except Exception as e:
            print(f"[오류] 처리 중 문제가 발생했습니다: {e}")
            continue

        print("\n" + "-" * 62)
        print(result["final_answer"])
        print("-" * 62)
        print("(결과를 저장하려면 '저장'을 입력하세요.)")

        last_result = result
