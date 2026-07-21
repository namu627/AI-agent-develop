def chatbot():
    print("🤖 챗봇에 오신 것을 환영합니다! (종료하려면 '종료'를 입력하세요)")
    print("-" * 50)

    while True:
        # 사용자 입력 받기 (양끝 공백 제거)
        user_input = input("\n나: ").strip()

        # 1. 종료 조건
        if user_input == "종료":
            print("챗봇: 대화를 종료합니다. 좋은 하루 보내세요! 👋")
            break

        # 2. 키워드별 응답 조건
        elif "안녕" in user_input:
            print("챗봇: 안녕하세요! 오늘 어떤 도움이 필요하신가요? 😊")

        elif "날씨" in user_input:
            print(
                "챗봇: 오늘은 창밖을 한번 보세요! 계절과 상관없이 좋은 날씨길 바랍니다. ☀️"
            )

        elif "도움말" in user_input:
            print("챗봇: [도움말]")
            print(" - '안녕': 인사 나누기")
            print(" - '날씨': 날씨 정보 물어보기")
            print(" - '종료': 프로그램 종료하기")

        # 3. 그 외 기본 응답
        else:
            print(
                "챗봇: 죄송해요, 무슨 말씀인지 잘 이해하지 못했어요. '도움말'을 입력해보세요!"
            )


# 챗봇 실행
if __name__ == "__main__":
    chatbot()