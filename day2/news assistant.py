"""
뉴스 검색·요약 에이전트
- Gemini API (gemini-2.5-flash) + NewsData API 연동
- 사용자가 입력한 키워드/카테고리/날짜범위/결과개수로 뉴스를 검색하고, LLM이 핵심 내용을 요약해서 응답
- 검색 기록은 search_history.json 파일에 누적 저장
"""

import os
import json
import re
from datetime import datetime

import requests
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------------
# 1. 환경 변수 로드
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
if not NEWSDATA_API_KEY:
    raise ValueError("NEWSDATA_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# 2. system_instruction 정의
# -----------------------------
SYSTEM_INSTRUCTION = """당신은 뉴스 요약 전문가입니다.
검색된 뉴스를 바탕으로 핵심 내용을 요약해주세요.
- 각 뉴스의 핵심 내용을 2~3문장으로 요약
- 중립적이고 객관적인 표현 사용
- 중요도 순으로 정렬해서 제시
- 출처(뉴스 제목)와 기사 날짜를 함께 표시"""

# -----------------------------
# 3. Gemini 모델 및 chat 세션 생성 (대화 기록 유지)
# -----------------------------
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION,
)

chat = model.start_chat(history=[])

NEWSDATA_URL = "https://newsdata.io/api/1/news"
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_history.json")

# NewsData API가 지원하는 카테고리 목록
CATEGORIES = [
    "business", "crime", "domestic", "education", "entertainment",
    "environment", "food", "health", "lifestyle", "politics",
    "science", "sports", "technology", "top", "tourism", "world", "other",
]


# -----------------------------
# 4. 뉴스 검색 함수 (Tool)
# -----------------------------
def search_news(
    keyword: str,
    category: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    num_results: int = 5,
) -> list[dict]:
    """
    NewsData API를 이용해 한국어 뉴스를 검색하고,
    [{"title": ..., "content": ..., "date": ...}, ...] 형태의 리스트를 반환한다.

    Args:
        keyword: 검색 키워드
        category: 뉴스 카테고리 (예: technology, sports 등). None이면 전체 카테고리.
        from_date: 검색 시작일 (YYYY-MM-DD). None이면 제한 없음.
        to_date: 검색 종료일 (YYYY-MM-DD). None이면 제한 없음.
        num_results: 반환할 최대 뉴스 개수

    API 호출 실패 시 예외를 발생시킨다.
    """
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": keyword,
        "language": "ko",
    }

    if category:
        params["category"] = category
    if from_date:
        params["from_date"] = from_date
    if to_date:
        params["to_date"] = to_date

    try:
        response = requests.get(NEWSDATA_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"NewsData API 호출 중 오류가 발생했습니다: {e}")

    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(f"NewsData API 응답 오류: {data.get('results', data)}")

    articles = data.get("results", []) or []

    news_list = []
    for article in articles[:num_results]:
        title = article.get("title") or "(제목 없음)"
        content = (
            article.get("content")
            or article.get("description")
            or "(본문 내용 없음)"
        )
        pub_date = article.get("pubDate") or "(날짜 정보 없음)"
        news_list.append({"title": title, "content": content, "date": pub_date})

    return news_list


# -----------------------------
# 5. 뉴스 목록을 LLM 입력용 텍스트로 변환
# -----------------------------
def format_news_for_prompt(keyword: str, news_list: list[dict]) -> str:
    if not news_list:
        return f"'{keyword}' 키워드로 검색된 뉴스가 없습니다."

    lines = [f"[검색 키워드: {keyword}]", "다음은 검색된 뉴스 목록입니다.\n"]
    for idx, news in enumerate(news_list, start=1):
        lines.append(f"{idx}. 제목: {news['title']}")
        lines.append(f"   날짜: {news['date']}")
        lines.append(f"   내용: {news['content']}\n")

    return "\n".join(lines)


# -----------------------------
# 6. 검색 기록 저장/조회 (Memory)
# -----------------------------
def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # 파일이 손상된 경우, 기존 기록을 잃지 않도록 백업 후 초기화
        return []


def save_history_entry(
    keyword: str,
    category: str | None,
    from_date: str | None,
    to_date: str | None,
    num_results: int,
    result_count: int,
) -> None:
    history = load_history()
    history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keyword": keyword,
            "category": category or "전체",
            "from_date": from_date or "-",
            "to_date": to_date or "-",
            "requested_count": num_results,
            "result_count": result_count,
        }
    )
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[경고] 검색 기록 저장에 실패했습니다: {e}")


def print_history() -> None:
    history = load_history()
    if not history:
        print("저장된 검색 기록이 없습니다.")
        return

    print("\n" + "=" * 50)
    print("검색 기록")
    print("=" * 50)
    for idx, entry in enumerate(history, start=1):
        print(
            f"{idx}. [{entry['timestamp']}] 키워드: {entry['keyword']} | "
            f"카테고리: {entry['category']} | "
            f"기간: {entry['from_date']} ~ {entry['to_date']} | "
            f"결과: {entry['result_count']}건 (요청 {entry['requested_count']}건)"
        )
    print("=" * 50)


# -----------------------------
# 7. 사용자 입력 도우미 함수
# -----------------------------
def prompt_category() -> str | None:
    print("\n카테고리를 선택하세요 (숫자 입력, 그냥 Enter 시 전체):")
    for idx, cat in enumerate(CATEGORIES, start=1):
        print(f"  {idx}. {cat}")

    choice = input("카테고리 번호: ").strip()
    if not choice:
        return None

    if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
        return CATEGORIES[int(choice) - 1]

    print("잘못된 입력입니다. 전체 카테고리로 검색합니다.")
    return None


def prompt_date(label: str) -> str | None:
    while True:
        value = input(f"{label} (YYYY-MM-DD, 그냥 Enter 시 제한 없음): ").strip()
        if not value:
            return None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return value
        print("날짜 형식이 올바르지 않습니다. 예: 2025-01-15")


def prompt_num_results() -> int:
    while True:
        value = input("검색 결과 개수 (1~10, 그냥 Enter 시 기본 5개): ").strip()
        if not value:
            return 5
        if value.isdigit() and 1 <= int(value) <= 10:
            return int(value)
        print("1에서 10 사이의 숫자를 입력해주세요.")


# -----------------------------
# 8. 메인 루프
# -----------------------------
def main():
    print("=" * 50)
    print("뉴스 검색·요약 에이전트")
    print("키워드를 입력하면 관련 뉴스를 검색하고 요약해드립니다.")
    print("'기록' 입력 시 지난 검색 기록을 볼 수 있습니다.")
    print("종료하려면 '종료'를 입력하세요.")
    print("=" * 50)

    while True:
        keyword = input("\n검색할 키워드를 입력하세요: ").strip()

        if keyword == "종료":
            print("에이전트를 종료합니다.")
            break

        if keyword == "기록":
            print_history()
            continue

        if not keyword:
            print("키워드를 입력해주세요.")
            continue

        # 검색 옵션 입력 (카테고리 / 날짜 범위 / 결과 개수)
        category = prompt_category()
        from_date = prompt_date("검색 시작일")
        to_date = prompt_date("검색 종료일")
        num_results = prompt_num_results()

        # 1) 뉴스 검색 (Tool 호출)
        try:
            news_list = search_news(
                keyword,
                category=category,
                from_date=from_date,
                to_date=to_date,
                num_results=num_results,
            )
        except RuntimeError as e:
            print(f"[뉴스 검색 실패] {e}")
            continue
        except Exception as e:
            print(f"[알 수 없는 오류] 뉴스 검색 중 문제가 발생했습니다: {e}")
            continue

        # 검색 기록 저장 (결과가 없어도 기록은 남김)
        save_history_entry(
            keyword, category, from_date, to_date, num_results, len(news_list)
        )

        if not news_list:
            print(f"'{keyword}'에 대한 검색 결과가 없습니다.")
            continue

        # 2) LLM에게 요약 요청 (chat 세션 사용 → 대화 기록 유지)
        prompt = format_news_for_prompt(keyword, news_list)

        try:
            response = chat.send_message(prompt)
            print("\n" + "=" * 50)
            print(response.text)
            print("=" * 50)
        except Exception as e:
            print(f"[Gemini API 호출 실패] 요약 생성 중 오류가 발생했습니다: {e}")
            continue


if __name__ == "__main__":
    main()