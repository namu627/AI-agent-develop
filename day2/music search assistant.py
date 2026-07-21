import os
import google.generativeai as genai
from google.generativeai.types import Tool
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY를 .env 파일에서 찾을 수 없습니다. 프로그램을 종료합니다.")
    exit(1)

genai.configure(api_key=api_key)

SYSTEM_INSTRUCTION = """당신은 음악 전문가입니다.
키워드가 가사에 포함된 음악을 찾아주세요.
-못찾으면 실패라고 출력
-여러개일 경우 최대 5가지만 출력
- 출처를 함께 표시

[중요 - 반드시 지킬 것]
- 반드시 google_search 도구를 사용해 실제 가사 정보를 검색한 뒤 답변하세요.
- 검색 결과에서 키워드가 실제로 가사에 포함된 것을 확인한 곡만 답변에 포함하세요.
- 검색으로 확인되지 않거나 확신이 서지 않는 곡은 절대로 지어내거나 추측해서 답하지 마세요.
- 검색 결과가 없거나 근거를 찾지 못하면 반드시 "실패"라고만 출력하세요.
- 모든 곡에는 검색으로 확인한 출처(웹 링크 또는 사이트명)를 함께 표시하세요."""

# google_search 도구를 연결해 실제 검색 결과에 근거해서 답변하도록 설정 (환각 방지)
search_tool = Tool(google_search=genai.protos.GoogleSearch())

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[search_tool]
)

# 대화 기록이 유지되도록 chat 세션 생성
chat = model.start_chat(history=[])

print("=== 음악 검색 에이전트 ===")
print("가사에 포함된 키워드로 음악을 찾아드립니다. ('종료' 입력 시 프로그램 종료)\n")

while True:
    user_input = input("사용자: ").strip()

    if user_input == "종료":
        print("음악 검색 에이전트를 종료합니다.")
        break

    if not user_input:
        print("키워드를 입력해주세요.")
        continue

    try:
        response = chat.send_message(user_input)
        print(f"\n에이전트:\n{response.text}")

        # 실제로 검색이 이루어졌는지, 어떤 출처를 참고했는지 확인용 출력
        try:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            if grounding_metadata and grounding_metadata.grounding_chunks:
                print("\n[참고한 검색 출처]")
                for chunk in grounding_metadata.grounding_chunks:
                    if chunk.web:
                        print(f"- {chunk.web.title}: {chunk.web.uri}")
        except (IndexError, AttributeError):
            pass

        print()
    except Exception as e:
        print(f"\n[오류] API 호출 중 문제가 발생했습니다: {e}\n")