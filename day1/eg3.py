def main():
    print("=== 사용자 정보 입력 ===")
    
    # 1. 사용자 정보 입력받기
    name = input("이름을 입력하세요: ").strip()
    age = input("나이를 입력하세요: ").strip()
    
    # 관심 분야는 쉼표로 구분하여 입력받기
    raw_interests = input("관심 분야를 입력하세요 (쉼표로 구분, 예: 파이썬, 게임, 파이썬): ")
    
    # 입력된 문자열을 쉼표 기준으로 나누고 공백 제거하여 리스트로 변환
    interest_list = [item.strip() for item in raw_interests.split(",") if item.strip()]

    # 2. 딕셔너리로 사용자 정보 저장
    user_info = {
        "이름": name,
        "나이": age,
        "관심분야": interest_list
    }

    # 3. 중복을 제거한 관심 분야 집합(set) 생성
    unique_interests = set(user_info["관심분야"])

    # 4. 결과 출력
    print("\n" + "=" * 25)
    print(" [ 사용자 정보 ]")
    print(f"• 이름: {user_info['이름']}")
    print(f"• 나이: {user_info['나이']}세")
    print("-" * 25)
    
    # 전체 관심 분야 출력 (리스트 형태)
    print(f"• 관심 분야 전체 (리스트): {user_info['관심분야']}")
    
    # 중복 제거된 관심 분야 출력 (집합 형태)
    print(f"• 중복 제거된 관심 분야 (집합): {unique_interests}")
    print("=" * 25)

if __name__ == "__main__":
    main()