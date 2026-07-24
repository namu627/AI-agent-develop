# app.py
import json
import os
import calendar
from datetime import date, datetime

import streamlit as st
from agent import run_planner, days_left, generate_study_schedule

st.set_page_config(page_title="AI 학습 플래너", page_icon="📚", layout="wide")

EXAM_FILE = "exams.json"
PLAN_FILE = "plans.json"
COLORS = ["#4C8BF5", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC",
          "#26C6DA", "#EC407A", "#8D6E63", "#5C6BC0", "#9CCC65"]


# ---------------- Storage ----------------
def load_exams():
    if os.path.exists(EXAM_FILE):
        try:
            with open(EXAM_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_exams(exams):
    with open(EXAM_FILE, "w", encoding="utf-8") as f:
        json.dump(exams, f, ensure_ascii=False, indent=2)


def load_plans():
    if os.path.exists(PLAN_FILE):
        try:
            with open(PLAN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_plans(plans):
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)


if "exams" not in st.session_state:
    st.session_state.exams = load_exams()
if "plans" not in st.session_state:
    st.session_state.plans = load_plans()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cal_year" not in st.session_state:
    st.session_state.cal_year = date.today().year
if "cal_month" not in st.session_state:
    st.session_state.cal_month = date.today().month


def color_of(name):
    names = [e["name"] for e in st.session_state.exams]
    return COLORS[names.index(name) % len(COLORS)] if name in names else COLORS[0]


def dtag(d):
    return f"D-{d}" if d > 0 else ("D-DAY" if d == 0 else f"D+{abs(d)}")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📝 시험 등록")
    with st.form("add_exam", clear_on_submit=True):
        name = st.text_input("시험 이름", placeholder="예: 정보처리기사 필기")
        subject = st.text_input("과목 / 분야", placeholder="예: 정보처리기사")
        exam_date = st.date_input("시험 날짜", value=date.today())
        submitted = st.form_submit_button("➕ 시험 추가", use_container_width=True)

    if submitted:
        if not name.strip() or not subject.strip():
            st.warning("시험 이름과 과목을 모두 입력해주세요.")
        elif any(e["name"] == name.strip() for e in st.session_state.exams):
            st.warning("같은 이름의 시험이 이미 있습니다.")
        else:
            st.session_state.exams.append({
                "name": name.strip(),
                "subject": subject.strip(),
                "exam_date": exam_date.strftime("%Y-%m-%d"),
            })
            st.session_state.exams.sort(key=lambda x: x["exam_date"])
            save_exams(st.session_state.exams)
            st.success(f"'{name}' 등록 완료")
            st.rerun()

    st.divider()
    st.header("📋 등록된 시험")

    if not st.session_state.exams:
        st.info("등록된 시험이 없습니다.")
    else:
        for i, e in enumerate(sorted(st.session_state.exams, key=lambda x: x["exam_date"])):
            d = days_left(e["exam_date"])
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"<div style='border-left:5px solid {color_of(e['name'])};"
                    f"padding:4px 0 4px 8px;margin-bottom:6px;'>"
                    f"<b>{e['name']}</b><br>"
                    f"<span style='font-size:0.8rem;color:gray;'>{e['subject']} · {e['exam_date']}</span><br>"
                    f"<span style='font-size:0.85rem;color:{'#EF5350' if 0 <= d <= 7 else '#4C8BF5'};'>"
                    f"<b>{dtag(d)}</b></span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑️", key=f"del_{e['name']}_{i}", help="삭제"):
                    st.session_state.exams = [
                        x for x in st.session_state.exams if x["name"] != e["name"]
                    ]
                    save_exams(st.session_state.exams)
                    st.rerun()

        if st.button("전체 삭제", use_container_width=True):
            st.session_state.exams = []
            save_exams([])
            st.rerun()

    st.divider()
    study_hours = st.slider("하루 학습 가능 시간 (시간)", 0.5, 12.0, 3.0, 0.5)

    upcoming = [e for e in st.session_state.exams if days_left(e["exam_date"]) >= 0]
    if upcoming:
        nearest = min(upcoming, key=lambda x: days_left(x["exam_date"]))
        st.metric("가장 임박한 시험", nearest["name"], dtag(days_left(nearest["exam_date"])))

    st.divider()
    st.header("📖 캘린더 학습 계획")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✨ 계획 생성", use_container_width=True):
            if not st.session_state.exams:
                st.warning("먼저 시험을 등록해주세요.")
            else:
                with st.spinner("등록된 시험으로 학습 계획을 만드는 중..."):
                    try:
                        schedule = generate_study_schedule(st.session_state.exams, study_hours)
                    except Exception as ex:
                        schedule = []
                        st.error(f"계획 생성 실패: {ex}")
                if schedule:
                    st.session_state.plans = schedule
                    save_plans(schedule)
                    st.success(f"{len(schedule)}개의 계획 항목이 캘린더에 반영되었습니다.")
                    st.rerun()
                else:
                    st.warning("계획을 생성하지 못했습니다. 다시 시도해주세요.")
    with c2:
        if st.button("🗑️ 계획 삭제", use_container_width=True):
            st.session_state.plans = []
            save_plans([])
            st.rerun()

    st.divider()
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- Calendar ----------------
def render_calendar(year, month):
    exam_map = {}
    for e in st.session_state.exams:
        exam_map.setdefault(e["exam_date"], []).append(e)

    plan_map = {}
    for p in st.session_state.plans:
        plan_map.setdefault(p.get("date"), []).append(p)

    cal = calendar.Calendar(firstweekday=6)  # 일요일 시작
    weeks = cal.monthdatescalendar(year, month)
    today = date.today()

    html = """<style>
    .cal{width:100%;border-collapse:collapse;table-layout:fixed;}
    .cal th{padding:8px;font-size:0.85rem;border-bottom:2px solid #ddd;}
    .cal td{height:110px;max-height:110px;vertical-align:top;border:1px solid #eee;padding:4px;overflow-y:auto;}
    .daynum{font-size:0.8rem;font-weight:600;}
    .out{color:#ccc;}
    .today{background:#FFF8E1;}
    .pill{display:block;font-size:0.68rem;color:#fff;border-radius:4px;
          padding:1px 4px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
    .pill.plan{color:#333;background:#fff!important;border:1px solid;}
    .pill.more{color:#888;background:#f0f0f0!important;border:none;text-align:center;}
    </style><table class='cal'><tr>"""
    for wd, col in zip(["일", "월", "화", "수", "목", "금", "토"],
                       ["#EF5350", "#333", "#333", "#333", "#333", "#333", "#4C8BF5"]):
        html += f"<th style='color:{col}'>{wd}</th>"
    html += "</tr>"

    for week in weeks:
        html += "<tr>"
        for day in week:
            classes = []
            if day.month != month:
                classes.append("out")
            if day == today:
                classes.append("today")
            html += f"<td class='{' '.join(classes)}'><span class='daynum'>{day.day}</span>"
            for e in exam_map.get(day.strftime("%Y-%m-%d"), []):
                html += (f"<span class='pill' style='background:{color_of(e['name'])}' "
                         f"title='{e['name']} ({e['subject']})'>📌 {e['name']}</span>")
            plan_items = plan_map.get(day.strftime("%Y-%m-%d"), [])
            for p in plan_items[:3]:
                c = color_of(p.get("exam_name", ""))
                p_title = f"{p.get('exam_name', '')} · {p.get('topic', '')} ({p.get('hours', '')}시간)"
                html += (f"<span class='pill plan' style='border-color:{c};color:{c}' "
                         f"title='{p_title}'>"
                         f"📖 {p.get('subject', '')} {p.get('hours', '')}h</span>")
            if len(plan_items) > 3:
                html += f"<span class='pill more'>+{len(plan_items) - 3}개 더</span>"
            html += "</td>"
        html += "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------- Main ----------------
st.title("📚 AI 학습 플래너")
st.caption("LangGraph Supervisor 패턴 · 여러 시험 통합 관리")

tab_chat, tab_cal = st.tabs(["💬 플래너 채팅", "📅 캘린더"])

with tab_cal:
    c1, c2, c3, c4 = st.columns([1, 1, 4, 1])
    with c1:
        if st.button("◀ 이전"):
            m, y = st.session_state.cal_month, st.session_state.cal_year
            st.session_state.cal_month, st.session_state.cal_year = (12, y - 1) if m == 1 else (m - 1, y)
            st.rerun()
    with c2:
        if st.button("다음 ▶"):
            m, y = st.session_state.cal_month, st.session_state.cal_year
            st.session_state.cal_month, st.session_state.cal_year = (1, y + 1) if m == 12 else (m + 1, y)
            st.rerun()
    with c3:
        st.subheader(f"{st.session_state.cal_year}년 {st.session_state.cal_month}월")
    with c4:
        if st.button("오늘"):
            st.session_state.cal_year = date.today().year
            st.session_state.cal_month = date.today().month
            st.rerun()

    render_calendar(st.session_state.cal_year, st.session_state.cal_month)

    if st.session_state.exams:
        st.divider()
        st.markdown("**범례** · 📌 시험일 &nbsp; 📖 학습 계획")
        legend = "".join(
            f"<span style='display:inline-block;background:{color_of(e['name'])};color:#fff;"
            f"border-radius:4px;padding:2px 8px;margin:2px;font-size:0.8rem;'>"
            f"{e['name']} · {e['exam_date']} ({dtag(days_left(e['exam_date']))})</span>"
            for e in sorted(st.session_state.exams, key=lambda x: x["exam_date"])
        )
        st.markdown(legend, unsafe_allow_html=True)

        if not st.session_state.plans:
            st.caption("아직 캘린더 학습 계획이 없습니다. 사이드바에서 '✨ 계획 생성'을 눌러보세요.")

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("무엇을 도와드릴까요? (예: 등록된 시험 전부 준비하는 계획 세워줘)"):
        if not st.session_state.exams:
            st.warning("사이드바에서 시험을 먼저 등록해주세요.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("에이전트가 분석 중입니다..."):
                try:
                    result = run_planner(
                        user_input=prompt,
                        exams=st.session_state.exams,
                        study_hours=study_hours,
                    )
                    answer = result["answer"]
                    next_agent = result.get("next_agent")
                except Exception as ex:
                    answer = f"오류가 발생했습니다: {ex}"
                    next_agent = None
            st.markdown(answer)

            schedule = []
            if next_agent == "plan_agent":
                with st.spinner("캘린더에 반영할 일정을 정리하는 중..."):
                    try:
                        schedule = generate_study_schedule(st.session_state.exams, study_hours)
                    except Exception:
                        schedule = []
                if schedule:
                    st.caption(f"📅 캘린더에 {len(schedule)}개 일정이 반영되었습니다. '📅 캘린더' 탭에서 확인하세요.")

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if schedule:
            st.session_state.plans = schedule
            save_plans(schedule)
            st.rerun()