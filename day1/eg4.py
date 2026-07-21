import sys

# 1. 인사 기능 함수
def greet():
    return "안녕하세요! 만나서 반가워요. 오늘 어떤 도움이 필요하신가요?"

# 2. 날씨 안내 기능 함수
def get_weather():
    # 실제 API 연동 전 임시 응답 처리
    return "오늘의 날씨는 대체로 화창하며, 산책하기 딱 좋은 기온입니다! ☀️"

# 3. 도움말 기능 함수
def get_help():
    return (
        "=== [ 도움말 ] ===\n"
        "다음과 같은 키워드를 포함해서 입력해 보세요!\n"
        "• 인사: '안녕', '반가워'\n"
        "• 날씨: '날씨', '비', '눈'\n"
        "• 도움말: '도움말', '메뉴', '기능'\n"
        "• 종료: '종료', '끝', 'exit'"
    )

# 4. 사용자 입력을 분석해 적절한 함수를 호출하는 반응 함수
def respond(user_input):
    # 대소문자 구분 없이 처리하기 위해 소문자 변환
    text = user_input.strip().lower()
    
    if any(keyword in text for keyword in ["안녕", "반가워", "hi", "hello"]):
        return greet()
    elif any(keyword in text for keyword in ["날씨", "비", "눈", "맑음"]):
        return get_weather()
    elif any(keyword in text for keyword in ["도움", "도움말", "기능", "메뉴", "help"]):
        return get_help()
    else:
        return "죄송해요, 무슨 말씀이신지 잘 이해하지 못했어요. '도움말'을 입력해 사용 가능한 기능을 확인해 보세요!"

# 5. 메인 실행 루프
def main():
    print("🤖 에이전트가 활성화되었습니다. ('도움말' 입력 시 기능 안내 / '종료' 입력 시 프로그램 종료)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n사용자: ")
            
            # 종료 조건 확인
            if user_input.strip().lower() in ["종료", "끝", "exit", "quit"]:
                print("에이전트: 이용해 주셔서 감사합니다. 프로그램을 종료합니다. 👋")
                break
            
            # 빈 입력 처리
            if not user_input.strip():
                continue
                
            # 응답 출력
            response = respond(user_input)
            print(f"에이전트: {response}")
            
        except (KeyboardInterrupt, EOFError):
            print("\n에이전트: 프로그램을 강제 종료합니다.")
            break

# 프로그램 시작점
if __name__ == "__main__":
    main()