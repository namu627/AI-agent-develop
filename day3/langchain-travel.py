# travel_planner.py
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ── Tool: 날씨 조회 ──────────────────────────────
_weather_cache = {}

@tool
def get_weather(city: str) -> str:
    """도시명(영문 권장)을 입력받아 현재 날씨와 향후 5일(3시간 간격) 예보를 반환합니다.
    여행 일정을 짤 때 반드시 이 도구로 날씨를 확인하고 일정에 반영하세요."""
    key = city.strip().lower()
    if key in _weather_cache:
        return _weather_cache[key]
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "kr",
        }
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return f"날씨 조회 실패: {res.status_code} - {res.text[:200]}"

        data = res.json()
        daily = {}
        for item in data["list"]:
            date = item["dt_txt"].split(" ")[0]
            daily.setdefault(date, []).append(item)

        weekdays = ["월", "화", "수", "목", "금", "토", "일"]
        lines = [f"[{data['city']['name']} 5일 예보]"]
        for date, items in list(daily.items())[:5]:
            temps = [i["main"]["temp"] for i in items]
            desc = items[len(items) // 2]["weather"][0]["description"]
            rain = any("rain" in i["weather"][0]["main"].lower() for i in items)
            wd = weekdays[datetime.strptime(date, "%Y-%m-%d").weekday()]
            lines.append(
                f"- {date}({wd}): {desc}, {min(temps):.1f}~{max(temps):.1f}°C"
                f"{' / 강수 예상 ☔' if rain else ''}"
            )
        lines.append("※ 무료 예보는 5일까지만 제공됩니다. 이후 날짜는 평년 기준으로 안내하세요.")
        result = "\n".join(lines)
        _weather_cache[key] = result
        return result
    except requests.exceptions.Timeout:
        return "날씨 API 요청 시간이 초과되었습니다."
    except Exception as e:
        return f"날씨 조회 중 오류 발생: {e}"


tools = [get_weather]

# ── 시스템 프롬프트 ──────────────────────────────
system_instruction = """당신은 10년 경력의 전문 여행 플래너입니다.
사용자의 여행 조건을 바탕으로 실용적이고 구체적인 여행 일정을 제안해주세요.
- 예산에 맞는 현실적인 일정 구성
- 이동 동선을 고려한 효율적인 코스 설계
- 숙박, 식비, 교통, 관광 예산을 구체적으로 배분
- 현지 맛집과 숨은 명소 포함"""

OUTPUT_RULE = """

[일정 작성 규칙]
1. get_weather 도구는 목적지당 딱 한 번만 호출하세요. 이미 조회한 도시는 이전 결과를 그대로 사용하고 절대 재호출하지 마세요.
2. 각 일차 제목 옆에 날짜와 날씨를 표기합니다.
   형식: ### 1일차 (10/03 금) | 🌧️ 비 · 17~21°C
   아이콘: 맑음 ☀️ / 구름 ⛅ / 흐림 ☁️ / 비 🌧️ / 눈 ❄️
   예보 범위(5일) 밖의 날짜는 '🗓️ 평년 기준 15~23°C'로 표기합니다.
   비 예보일에는 실내 명소를 배치하고, 맑은 날에는 야외 명소를 배치하세요.
3. 각 일차 끝에 그날 지출 소계를 한 줄로 적습니다.
   형식: 💰 1일차 소계: 87,000원
4. 모든 일차가 끝나면 반드시 아래 형식의 총 예산표를 마지막에 출력하세요.

## 💳 총 예산 정리
| 항목 | 금액 |
|---|---|
| 항공/교통(왕복) | 000,000원 |
| 숙박 (0박) | 000,000원 |
| 식비 | 000,000원 |
| 현지 교통 | 000,000원 |
| 관광/입장료 | 000,000원 |
| 예비비 | 000,000원 |
| **총합계** | **000,000원** |

- 1인 기준 총액과 설정 예산 대비 잔액(또는 초과액)을 한 줄로 명시하세요.
- 항목별 금액의 합이 총합계와 정확히 일치해야 합니다.
"""

STYLE_GUIDE = {
    "1": ("힐링", "여유로운 일정. 하루 2~3곳 이내, 자연·온천·카페·산책 중심. 이동 시간 최소화, 충분한 휴식 시간 확보."),
    "2": ("액티브", "밀도 높은 일정. 하루 4~5곳, 액티비티·트레킹·수상스포츠·야경 명소 포함. 이른 출발과 효율적 동선 강조."),
    "3": ("미식", "식사 중심 일정. 끼니마다 현지 맛집·시장·로컬 술집을 지정하고, 예약 필요 여부와 대표 메뉴·가격대를 명시."),
}

def looks_like_plan(text: str) -> bool:
    """응답이 여행 일정표인지 판단"""
    has_day = any(k in text for k in ["일차", "Day 1", "DAY 1", "첫째 날"])
    has_total = any(k in text for k in ["총합계", "총 예산", "총액"])
    return has_day and has_total and len(text) > 300

def choose_style():
    print("\n여행 스타일을 선택하세요.")
    for k, (name, desc) in STYLE_GUIDE.items():
        print(f"  {k}. {name} 여행 - {desc.split('.')[0]}")
    while True:
        sel = input("번호 입력 (1/2/3): ").strip()
        if sel in STYLE_GUIDE:
            name, desc = STYLE_GUIDE[sel]
            print(f"→ '{name} 여행' 스타일로 진행합니다.\n")
            return f"\n\n[선택된 여행 스타일: {name}]\n{desc}\n모든 일정은 이 스타일에 맞춰 구성하세요."
        print("1, 2, 3 중에서 선택해주세요.")


def looks_like_plan(text: str) -> bool:
    """응답이 여행 일정표인지 판단"""
    keywords = ["1일차", "Day 1", "DAY 1", "첫째 날", "일차"]
    budget_keywords = ["예산", "숙박", "총 비용", "원"]
    has_day = any(k in text for k in keywords)
    has_budget = any(k in text for k in budget_keywords)
    return has_day and has_budget and len(text) > 300


def save_to_file(text, city_hint=""):
    answer = input("\n이 일정을 txt 파일로 저장할까요? (y/n): ").strip().lower()
    if answer not in ("y", "yes", "예", "ㅇ"):
        print("저장하지 않았습니다.")
        return

    default = f"travel_plan_{city_hint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt".replace("__", "_")
    name = input(f"파일명 (엔터 시 '{default}'): ").strip()
    filename = name if name else default
    if not filename.endswith(".txt"):
        filename += ".txt"

    header = (
        "=" * 50 + "\n"
        f"여행 일정표\n"
        f"작성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        + "=" * 50 + "\n\n"
    )
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header + text + "\n")
        print(f"저장 완료: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"저장 실패: {e}")


def main():
    if not OPENAI_API_KEY:
        print(".env 파일에 OPENAI_API_KEY가 없습니다.")
        return
    if not OPENWEATHER_API_KEY:
        print("경고: OPENWEATHER_API_KEY가 없어 날씨 조회가 동작하지 않습니다.")

    style_prompt = choose_style()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction + style_prompt + OUTPUT_RULE),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

    print("=" * 60)
    print("여행 플래너 에이전트입니다. (종료하려면 '종료' 입력)")
    print("예: '10월 3박 4일 오사카, 예산 120만원으로 일정 짜줘'")
    print("=" * 60)

    last_output = ""
    while True:
        user_input = input("\n나: ").strip()
        if not user_input:
            continue
        if user_input == "종료":
            print("여행 플래너를 종료합니다. 즐거운 여행 되세요!")
            break
        if user_input == "저장":
            if last_output:
                save_to_file(last_output)
            else:
                print("저장할 일정이 없습니다.")
            continue
        if user_input == "스타일변경":
            style_prompt = choose_style()
            prompt.messages[0].prompt.template = system_instruction + style_prompt + OUTPUT_RULE
            continue

        try:
            result = executor.invoke({"input": user_input})
            last_output = result["output"]
            print(f"\n플래너: {last_output}")

            # 일정이 완성되면 자동으로 저장 여부 질문
            if looks_like_plan(last_output):
                print("\n[일정이 완성된 것 같습니다]")
                save_to_file(last_output)
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("잠시 후 다시 시도해주세요.")


if __name__ == "__main__":
    main()