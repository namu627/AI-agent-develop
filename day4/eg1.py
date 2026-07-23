from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI 
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str] # 대화 기록
    next_action: str # 다음 실행할 행동 ("tool" 또는 "end")
    tool_result: str # Tool 실행 결과
    llm= ChatOpenAI(model="gpt-4o-mini")

def think_node(state: AgentState) -> AgentState:
    """Thought: LLM이 상황을 판단하고 다음 행동을 결정하는 노드"""
    response = llm.invoke(state["messages"])
    # LLM 응답에 Tool 호출이 필요하면 "tool", 완료면 "end"
    state["next_action"] = "tool" if "검색" in response.content else "end"
    state["messages"].append(response.content)
    return state

def act_node(state: AgentState) -> AgentState:
    """Action: Tool을 실행하는 노드"""
    # Tool 실행 후 결과를 상태에 저장
    state["tool_result"] = "Tool 실행 결과"
    state["messages"].append(f"Observation: {state['tool_result']}")
    return state

#관찰자
def should_continue(state: AgentState) -> str:
    """Observation: Tool 결과를 보고 반복할지 종료할지 결정"""
    return state["next_action"]
    # "tool" → act_node로 돌아가 반복
    # "end" → 그래프 종료

# 그래프 구성
graph = StateGraph(AgentState)
graph.add_node("think", think_node)
graph.add_node("act", act_node)
graph.set_entry_point("think")
graph.add_conditional_edges("think", should_continue, {
"tool": "act", # Tool 실행 필요 → act_node로
"end": END # 완료 → 그래프 종료
})
graph.add_edge("act", "think")
# ↑ Tool 실행 후 다시 think_node로 돌아와 반복
# act_node 완료 후 다시 think_node로 돌아오는 순환 구조
# 이 한 줄이 ReAct 패턴의 "반복" 을 구현하는 부분
app = graph.compile()