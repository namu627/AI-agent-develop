import os
import requests
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY를 찾을 수 없습니다. 프로젝트 루트에 .env 파일을 만들고 "
        "GEMINI_API_KEY=발급받은_키 형식으로 저장해주세요."
    )

MODEL_NAME = "gemini-2.5-flash"
API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent?key={API_KEY}"
)

SYSTEM_INSTRUCTION = (
    "당신은 MBTI 전문가이자 콘텐츠 큐레이터입니다. 사용자의 MBTI유형의 연애 스타일 특징들을 알려주세요."
    "사용자의 MBTI 유형에 맞는 영화, 책, 음악을 추천해주세요. 추천 시 다음 원칙을 따르세요:" 
    "각 MBTI 유형의 성격 특성을 반영한 추천 - 추천 이유를 MBTI 특성과 연결해서 설명 - 친근하고 공감 가는 말투 사용 - 콘텐츠별로 최소 2개 이상 추천"
    "사용자의 mbti를 듣고 이후부턴 사용자를 부를땐 성격 유형으로 부를 것(ex. 자유로운 예술가, 모험을 즐기는 사업가)"
)


# MBTI 간이 검사용 질문
# 각 항목: (지표, 질문, 1번 선택지, 2번 선택지)
MBTI_TEST_QUESTIONS = [
    ("EI", "사람들과 함께 어울릴 때 에너지가 충전되는 편인가요?",
        "네, 사람들과 함께 있을 때 활력이 생겨요 (E)",
        "아니요, 혼자만의 시간이 있어야 재충전돼요 (I)"),
    ("SN", "새로운 것을 배울 때 어느 쪽에 더 끌리나요?",
        "구체적인 사실과 경험, 현실적인 정보 (S)",
        "아이디어, 가능성, 큰 그림과 상상 (N)"),
    ("TF", "중요한 결정을 내릴 때 무엇을 더 우선하나요?",
        "논리적인 분석과 원칙, 객관적 기준 (T)",
        "사람들과의 관계, 감정과 상황에 대한 공감 (F)"),
    ("JP", "평소 생활 방식에 더 가까운 쪽은?",
        "계획을 세우고 미리 정리해두는 게 편해요 (J)",
        "그때그때 상황에 맞춰 유연하게 움직이는 게 편해요 (P)"),
]

MBTI_TEST_TRIGGERS = {"mbti검사", "mbti 검사", "검사", "mbti테스트", "mbti 테스트", "테스트"}


def run_mbti_test() -> str:
    """간단한 4문항 MBTI 자가진단을 진행하고 4글자 유형 문자열을 반환한다."""
    print("\n[간이 MBTI 검사를 시작할게요! 각 질문에 1 또는 2로 답해주세요.]\n")

    result = ""
    for dichotomy, question, option1, option2 in MBTI_TEST_QUESTIONS:
        while True:
            print(f"Q. {question}")
            print(f"  1) {option1}")
            print(f"  2) {option2}")
            answer = input("답변 (1/2): ").strip()

            if answer == "1":
                result += dichotomy[0]
                break
            elif answer == "2":
                result += dichotomy[1]
                break
            else:
                print("1 또는 2로만 답해주세요.\n")

        print()

    print(f"검사 결과, 당신의 MBTI 유형은 [{result}]로 예상돼요!\n")
    return result


def send_message(history: list) -> str:
    """
    history: [{"role": "user"|"model", "parts": [{"text": "..."}]}, ...]
    현재까지의 전체 대화 기록(history)을 매번 함께 전송해 문맥(대화 기록)을 유지한다.
    """
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": history,
    }

    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    # 응답에서 텍스트 추출
    return data["candidates"][0]["content"]["parts"][0]["text"]


def main():
    history = []  # 대화 기록을 직접 리스트로 관리 (chat 세션 대체)

    print("=" * 50)
    print("  MBTI 기반 콘텐츠 추천 에이전트")
    print("=" * 50)
    print("MBTI 유형과 원하는 콘텐츠(영화/책/음악)를 함께 입력해보세요.")
    print("예) 'INFP인데 영화 추천해줘', 'ENTJ 책 추천'")
    print("MBTI를 모른다면 'MBTI검사'라고 입력해보세요!")
    print("대화를 끝내려면 '종료'를 입력하세요.\n")

    while True:
        user_input = input("나: ").strip()

        if user_input == "종료":
            print("\n에이전트: 즐거운 대화였어요! 다음에 또 만나요 :)")
            break

        if not user_input:
            continue

        # MBTI 간이 검사 트리거
        if user_input.lower() in MBTI_TEST_TRIGGERS:
            mbti_result = run_mbti_test()
            user_input = (
                f"방금 간이 검사로 내 MBTI가 {mbti_result}로 나왔어. "
            )
            print(f"나: {user_input}")

        # 사용자 메시지를 기록에 추가
        history.append({"role": "user", "parts": [{"text": user_input}]})

        try:
            reply_text = send_message(history)
            print(f"\n에이전트: {reply_text}\n")

            # 모델 응답도 기록에 추가해야 다음 턴에서 문맥이 유지됨
            history.append({"role": "model", "parts": [{"text": reply_text}]})

        except requests.exceptions.HTTPError as e:
            print(f"\n[오류] API 요청 실패: {e}\n{e.response.text}\n")
            history.pop()  # 실패한 사용자 메시지는 기록에서 제거
        except Exception as e:
            print(f"\n[오류] 메시지 전송 중 문제가 발생했습니다: {e}\n")
            history.pop()


if __name__ == "__main__":
    main()