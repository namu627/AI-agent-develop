import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. .env 파일에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

genai.configure(api_key=API_KEY)

# 2. AI의 역할(system_instruction) 설정 - 자유롭게 수정 가능
SYSTEM_INSTRUCTION = (
    "당신은 친절하고 전문적인 15년차 시계 전문가입니다. "
    "사용자의 상황에 맞는 시계 구매 계획, 시계 추천, 예산 팁 등을 구체적이고 실용적으로 안내해주세요."
)

MODEL_NAME = "gemini-2.5-flash"


def main():
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        # 3. 대화 기록 유지를 위한 chat 세션 시작
        chat = model.start_chat(history=[])
    except Exception as e:
        print(f"모델 초기화 중 오류가 발생했습니다: {e}")
        return

    print("=== AI 어시스턴트 시작 (종료하려면 '종료' 입력) ===")

    # 4. while True 루프로 대화 이어가기
    while True:
        user_input = input("\n나: ").strip()

        if user_input == "종료":
            print("AI 어시스턴트를 종료합니다.")
            break

        if not user_input:
            continue

        # 5. API 호출 실패 시 예외처리
        try:
            response = chat.send_message(user_input)

            # 토큰 사용량 표시
            usage = response.usage_metadata
            if usage:
                print(
                    f"[토큰 사용량] 입력: {usage.prompt_token_count} / "
                    f"출력: {usage.candidates_token_count} / "
                    f"총합: {usage.total_token_count}"
                )

            print(f"AI: {response.text}")
        except Exception as e:
            print(f"오류가 발생했습니다. 다시 시도해주세요. ({e})")


if __name__ == "__main__":
    main()