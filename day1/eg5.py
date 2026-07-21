class ChatBot:

    def __init__(self, name: str):
        """챗봇의 이름과 대화 기록 리스트를 초기화합니다."""
        self.name = name
        self.history = []

    def respond(self, user_input: str) -> str:
        """사용자의 입력을 받아 처리하고 응답을 반환합니다."""
        # 1. 예외 처리: 입력값이 없거나 공백만 있는 경우
        if not user_input or not user_input.strip():
            raise ValueError("빈 문자열은 입력할 수 없습니다. 내용을 입력해주세요.")

        cleaned_input = user_input.strip()

        # 2. 간단한 규칙 기반 응답 처리
        if "안녕" in cleaned_input:
            bot_response = f"안녕하세요! 저는 {self.name}입니다. 무엇을 도와드릴까요?"
        elif "이름" in cleaned_input:
            bot_response = f"제 이름은 {self.name}입니다."
        elif "날씨" in cleaned_input:
            bot_response = "오늘 날씨는 정말 화창하네요!"
        else:
            bot_response = f"'{cleaned_input}'에 대해 말씀하셨군요. 더 자세히 이야기해 주세요!"

        # 3. 대화 기록 저장
        self.history.append((cleaned_input, bot_response))
        return bot_response

    def show_history(self):
        """저장된 대화 기록을 보기 쉽게 출력합니다."""
        print(f"\n=== {self.name}과의 대화 기록 ===")
        if not self.history:
            print("저장된 대화 기록이 없습니다.")
            return

        for idx, (user, bot) in enumerate(self.history, 1):
            print(f"[{idx}] 사용자: {user}")
            print(f"    챗봇({self.name}): {bot}")
        print("=" * 30 + "\n")


# --------------------------------------------------
# 실행 루프 (Main Execution)
# --------------------------------------------------
if __name__ == "__main__":
    # 챗봇 객체 생성
    bot = ChatBot(name="알파")
    print(f"🤖 {bot.name} 챗봇이 시작되었습니다! (종료하려면 '종료' 또는 'exit' 입력)")
    print("기록을 보고 싶다면 '기록'을 입력하세요.\n")

    while True:
        try:
            user_input = input("사용자: ")

            # 대화 종료 조건
            if user_input.strip().lower() in ["종료", "exit", "quit"]:
                print(f"🤖 {bot.name}: 대화를 종료합니다. 좋은 하루 되세요!")
                break

            # 대화 기록 출력 명령어
            if user_input.strip() == "기록":
                bot.show_history()
                continue

            # 챗봇 응답 생성 및 출력
            response = bot.respond(user_input)
            print(f"🤖 {bot.name}: {response}\n")

        except ValueError as e:
            # 예외 발생 시 에러 메시지 출력 후 대화 계속 진행
            print(f"⚠️ [입력 오류]: {e}\n")
        except KeyboardInterrupt:
            # Ctrl+C 입력 시 강제 종료 처리
            print(f"\n🤖 {bot.name}: 대화가 강제 종료되었습니다.")
            break