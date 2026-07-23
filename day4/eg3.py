from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str # 사용자 질문 —그래프 시작 시 입력
    retrieved_docs: str # 검색된 문서 내용 —retrieve_node가 채워줌
    answer: str # 최종 답변 —generate_node가 채워줌

def retrieve_node(state: RAGState) -> RAGState:
    """벡터 데이터베이스에서 관련 문서를 검색하는 노드"""
    # 사용자 질문을 벡터로 변환해서 유사한 chunk 검색
    docs = retriever.invoke(state["question"])
    # 검색된 chunk들을 줄바꿈으로 연결해 하나의 문자열로 합침
    # → LLM에 전달하기 쉬운 형태로 변환
    state["retrieved_docs"] = "\n".join([doc.page_contentfor doc in docs])
    return state

def generate_node(state: RAGState) -> RAGState:
    """검색된 문서를 바탕으로 LLM이 답변을 생성하는 노드"""
    # 검색된 문서 내용과 사용자 질문을 함께 프롬프트에 포함
    # → LLM이 채용공고 내용을 참고해서 정확한 답변을 생성할 수 있음
    prompt = f"""
    다음 채용공고 내용을 참고해서 질문에 답변해줘.
    채용공고 내용:
    {state["retrieved_docs"]}
    # ↑ retrieve_node가 검색한 관련 chunk들이 여기에 삽입됨
    질문: {state["question"]}
    """
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state

# 그래프 구성
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
# ↑ 항상 retrieve_node에서 시작 —먼저 문서를 검색한 뒤 답변 생성
graph.add_edge("retrieve", "generate")
# ↑ 검색 완료 후 항상 답변 생성 노드로 이동 (조건 없는 일반 엣지)
# → RAG는 항상 검색 후 생성 순서를 따르므로 조건 엣지가 필요 없음
graph.add_edge("generate", END)
# ↑ 답변 생성 완료 후 그래프 종료
app = graph.compile()