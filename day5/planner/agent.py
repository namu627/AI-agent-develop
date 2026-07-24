# agent.py
import os
import json
import requests
from typing import TypedDict, Optional, List, Dict
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)


class PlannerState(TypedDict, total=False):
    user_input: str
    exams: List[Dict]          # [{"name","subject","exam_date"}, ...]
    study_hours: float
    next_agent: str
    plan_result: Optional[str]
    news_result: Optional[str]
    progress_result: Optional[str]
    final_answer: Optional[str]


def days_left(exam_date: str) -> int:
    try:
        d = datetime.strptime(exam_date, "%Y-%m-%d").date()
        return (d - datetime.now().date()).days
    except Exception:
        return 0


def _exam_lines(exams: List[Dict]) -> str:
    if not exams:
        return "(등록된 시험 없음)"
    rows = []
    for e in sorted(exams, key=lambda x: x["exam_date"]):
        d = days_left(e["exam_date"])
        tag = f"D-{d}" if d > 0 else ("D-DAY" if d == 0 else f"D+{abs(d)}")
        rows.append(f"- {e['name']} / 과목: {e['subject']} / 시험일: {e['exam_date']} ({tag})")
    return "\n".join(rows)


def _subjects(exams: List[Dict]) -> str:
    return ", ".join(sorted({e["subject"] for e in exams})) or "일반 학습"


# ---------------- Supervisor ----------------
def supervisor_node(state: PlannerState) -> PlannerState:
    prompt = f"""당신은 학습 플래너 시스템의 슈퍼바이저입니다.
사용자 요청을 읽고 아래 세 에이전트 중 하나를 선택하세요.

- plan_agent: 학습 계획, 일정, 커리큘럼, 여러 시험 동시 준비 스케줄 요청
- news_agent: 최신 학습 자료, 뉴스, 트렌드, 참고 자료 검색 요청
- progress_agent: 오늘 공부한 내용 보고, 진도 점검, 피드백 요청

반드시 JSON만 출력: {{"next_agent": "plan_agent"}}

사용자 요청: {state['user_input']}"""

    res = llm.invoke([SystemMessage(content="너는 라우팅만 담당한다. JSON만 출력."),
                      HumanMessage(content=prompt)]).content.strip()
    res = res.replace("```json", "").replace("```", "").strip()
    try:
        next_agent = json.loads(res).get("next_agent", "plan_agent")
    except Exception:
        next_agent = "plan_agent"

    if next_agent not in ("plan_agent", "news_agent", "progress_agent"):
        next_agent = "plan_agent"
    return {**state, "next_agent": next_agent}


def route(state: PlannerState) -> str:
    return state.get("next_agent", "plan_agent")


# ---------------- Agents ----------------
def plan_agent_node(state: PlannerState) -> PlannerState:
    exams = state.get("exams", [])
    hours = state.get("study_hours", 3)

    prompt = f"""아래 등록된 모든 시험을 함께 대비하는 통합 학습 계획을 세워주세요.

[등록된 시험 목록]
{_exam_lines(exams)}

[하루 총 학습 가능 시간] {hours}시간
[사용자 요청] {state.get('user_input')}

요구사항:
1. 전체 전략 요약 (시험이 여러 개면 우선순위 근거를 명시: 임박도 · 난이도 · 분량)
2. 시험별 시간 배분 비율(%)과 그 이유 — 시험일이 가까울수록 비중을 높일 것
3. 주차별 로드맵 (주차 | 시험별 집중 내용 | 주요 목표)
4. 일별 학습 계획 표 (날짜 | 시험/과목 | 학습 주제 | 배분 시간)
   - 하루 총합이 반드시 {hours}시간을 넘지 않게 배분
   - 하루에 여러 시험을 다룰 경우 시간대를 나눠서 표기
5. 각 시험 직전 2~3일은 해당 시험 복습/모의고사에 집중 배치
6. 이미 지난 시험(D+)은 계획에서 제외하고 그 사실을 한 줄로 알려줄 것"""

    out = llm.invoke([SystemMessage(content="당신은 다중 시험 대비 학습 계획 전문가입니다. 한국어로 답변하세요."),
                      HumanMessage(content=prompt)]).content
    return {**state, "plan_result": out}


def generate_study_schedule(exams: List[Dict], study_hours: float) -> List[Dict]:
    """등록된 시험들을 기반으로 캘린더에 표시할 날짜별 학습 스케줄(JSON)을 생성한다."""
    valid_exams = [e for e in exams if days_left(e["exam_date"]) >= 0]
    if not valid_exams:
        return []

    prompt = f"""아래 등록된 시험들을 대비하는 일별 학습 스케줄을 JSON 배열로 생성하세요.

[등록된 시험 목록]
{_exam_lines(valid_exams)}

[하루 총 학습 가능 시간] {study_hours}시간

요구사항:
- 오늘부터 각 시험일 전날까지, 시험일마다 매일 한 개 이상의 항목을 생성 (시험일 이후는 제외)
- 하루에 여러 시험을 다룰 경우 항목을 여러 개로 나눌 것
- 하루 배분 시간 합이 {study_hours}시간을 넘지 않게 할 것
- 시험일이 가까울수록 해당 시험 비중을 높일 것
- 각 시험 직전 2~3일은 복습/모의고사 위주로 편성

반드시 아래 형식의 JSON 배열만 출력하세요 (설명, 코드블록 없이 순수 JSON):
[{{"date": "YYYY-MM-DD", "exam_name": "...", "subject": "...", "topic": "...", "hours": 2}}]"""

    res = llm.invoke([
        SystemMessage(content="당신은 학습 스케줄러입니다. 반드시 유효한 JSON 배열만 출력하고 다른 텍스트는 포함하지 마세요."),
        HumanMessage(content=prompt),
    ]).content.strip()

    res = res.replace("```json", "").replace("```", "").strip()
    try:
        schedule = json.loads(res)
        if isinstance(schedule, list):
            return [item for item in schedule if isinstance(item, dict) and item.get("date")]
    except Exception:
        pass
    return []


def news_agent_node(state: PlannerState) -> PlannerState:
    exams = state.get("exams", [])
    subjects = sorted({e["subject"] for e in exams})[:3] or ["학습"]

    articles_text = ""
    for subj in subjects:
        articles_text += f"\n■ [{subj}] 검색 결과\n"
        try:
            resp = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": NEWSDATA_API_KEY, "q": subj, "language": "ko"},
                timeout=15,
            )
            results = resp.json().get("results", [])[:3]
            if not results:
                resp = requests.get(
                    "https://newsdata.io/api/1/latest",
                    params={"apikey": NEWSDATA_API_KEY, "q": subj, "language": "en"},
                    timeout=15,
                )
                results = resp.json().get("results", [])[:3]

            if not results:
                articles_text += "  (검색 결과 없음)\n"
            for i, a in enumerate(results, 1):
                articles_text += (
                    f"  {i}. 제목: {a.get('title')}\n"
                    f"     링크: {a.get('link')}\n"
                    f"     요약: {(a.get('description') or '')[:250]}\n"
                )
        except Exception as e:
            articles_text += f"  (API 호출 실패: {e})\n"

    prompt = f"""등록된 시험 과목별 최신 자료 검색 결과입니다.

[등록된 시험]
{_exam_lines(exams)}

[검색 결과]
{articles_text}

요구사항:
1. 과목별로 섹션을 나눠 자료를 한 줄씩 정리하고 링크 포함
2. 각 자료를 해당 시험 준비에 어떻게 활용할지 제안
3. 검색 결과가 없는 과목은 공식 문서 · 인강 · 기출문제집 등 대체 자료를 추천"""

    out = llm.invoke([SystemMessage(content="당신은 학습 자료 큐레이터입니다. 한국어로 답변하세요."),
                      HumanMessage(content=prompt)]).content
    return {**state, "news_result": out}


def progress_agent_node(state: PlannerState) -> PlannerState:
    prompt = f"""학생의 오늘 학습 보고입니다.

[등록된 시험 목록]
{_exam_lines(state.get('exams', []))}

[하루 목표 학습 시간] {state.get('study_hours')}시간
[오늘 학습한 내용] {state.get('user_input')}

요구사항:
1. 오늘 학습 내용이 어느 시험에 해당하는지 매칭
2. 목표 대비 달성률(추정 %) 및 시험별 진도 균형 평가 (한쪽에 치우쳤다면 지적)
3. 잘한 점 / 보완할 점
4. 내일 해야 할 학습 3가지 (시험명 명시, 임박한 시험 우선)
5. 격려 메시지 한 줄"""

    out = llm.invoke([SystemMessage(content="당신은 학습 코치입니다. 한국어로 답변하세요."),
                      HumanMessage(content=prompt)]).content
    return {**state, "progress_result": out}


def final_node(state: PlannerState) -> PlannerState:
    exams = state.get("exams", [])
    header = (
        f"📚 **등록 시험 {len(exams)}개** | 하루 {state.get('study_hours')}시간\n\n"
        f"{_exam_lines(exams)}\n\n---\n\n"
    )

    parts = []
    if state.get("plan_result"):
        parts.append("### 🗓️ 통합 학습 계획\n" + state["plan_result"])
    if state.get("news_result"):
        parts.append("### 📰 최신 학습 자료\n" + state["news_result"])
    if state.get("progress_result"):
        parts.append("### ✅ 진도 체크 & 피드백\n" + state["progress_result"])

    body = "\n\n---\n\n".join(parts) if parts else "처리된 결과가 없습니다."
    return {**state, "final_answer": header + body}


# ---------------- Graph ----------------
def build_graph():
    g = StateGraph(PlannerState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("plan_agent", plan_agent_node)
    g.add_node("news_agent", news_agent_node)
    g.add_node("progress_agent", progress_agent_node)
    g.add_node("final", final_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route, {
        "plan_agent": "plan_agent",
        "news_agent": "news_agent",
        "progress_agent": "progress_agent",
    })
    g.add_edge("plan_agent", "final")
    g.add_edge("news_agent", "final")
    g.add_edge("progress_agent", "final")
    g.add_edge("final", END)
    return g.compile()


graph = build_graph()


def run_planner(user_input: str, exams: List[Dict], study_hours: float) -> Dict:
    state: PlannerState = {
        "user_input": user_input,
        "exams": exams or [],
        "study_hours": study_hours,
        "next_agent": "",
        "plan_result": None,
        "news_result": None,
        "progress_result": None,
        "final_answer": None,
    }
    result = graph.invoke(state)
    return {
        "answer": result.get("final_answer", "결과를 생성하지 못했습니다."),
        "next_agent": result.get("next_agent"),
    }


if __name__ == "__main__":
    demo = [
        {"name": "정보처리기사 필기", "subject": "정보처리기사", "exam_date": "2026-08-20"},
        {"name": "토익 정기시험", "subject": "TOEIC", "exam_date": "2026-09-13"},
    ]
    print(run_planner("두 시험 같이 준비하는 계획 세워줘", demo, 4)["answer"])