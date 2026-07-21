# 1. 사용자로부터 이름, 나이, 전공 입력받기
name = input("이름을 입력하세요: ")
age_input = input("나이를 입력하세요: ")
major = input("전공을 입력하세요: ")

# 2. 나이를 정수(int)형으로 변환하여 내년 나이 계산하기
age = int(age_input)
next_year_age = age + 1

# 3. f-string을 활용한 자기소개 출력
print("\n--- [ 자기소개서 ] ---")
print(
    f"안녕하세요! 제 이름은 {name}입니다. "
    f"전공은 {major}이며, 현재 나이는 {age}살(내년에는 {next_year_age}살)입니다."
)