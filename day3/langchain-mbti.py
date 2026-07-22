"""
MBTI 콘텐츠 추천 체인
--------------------
LangChain을 활용하여 사용자의 MBTI 유형과 원하는 콘텐츠 유형(영화/책/음악)을
입력받아, gpt-4o-mini 모델로 추천 목록을 생성하는 프로그램.

실행 환경: Miniconda 가상환경(Python 3.11), VSCode 터미널

사전 준비:
1. 아래 패키지 설치
   pip install langchain langchain-openai python-dotenv

2. 프로젝트 폴더에 .env 파일을 만들고 아래처럼 API 키 저장
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser


# 1. .env 파일에서 OpenAI API Key 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY를 찾을 수 없습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    )


# 2. 출력 파서 (콤마로 구분된 문자열 -> 리스트 변환)
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()


# 3. 프롬프트 템플릿 작성 (MBTI 유형 + 콘텐츠 유형 입력)
prompt = PromptTemplate(
    template=(
        "당신은 사용자의 성향에 맞는 콘텐츠를 추천하는 전문가입니다.\n"
        "MBTI 유형: {mbti}\n"
        "콘텐츠 유형: {content_type}\n\n"
        "위 MBTI 유형의 성향을 고려하여 어울리는 {content_type} 5가지를 추천해주세요.\n"
        "{format_instructions}"
    ),
    input_variables=["mbti", "content_type"],
    partial_variables={"format_instructions": format_instructions},
)


# 4. LLM 설정 (gpt-4o-mini 모델 사용)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key,
)


# 5. 체인 연결: PromptTemplate -> LLM -> OutputParser (LCEL 방식)
chain = prompt | llm | output_parser


def get_recommendation(mbti: str, content_type: str) -> list:
    """MBTI와 콘텐츠 유형을 받아 추천 리스트를 반환하는 함수"""
    result = chain.invoke({"mbti": mbti, "content_type": content_type})
    return result


def main():
    print("=" * 50)
    print("MBTI 콘텐츠 추천 프로그램")
    print("(종료하려면 'MBTI 입력' 단계에서 '종료'를 입력하세요)")
    print("=" * 50)

    valid_content_types = ["영화", "책", "음악"]

    # 실제 존재하는 16가지 MBTI 유형 목록
    valid_mbti_types = {
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP",
    }

    while True:
        mbti = input("\nMBTI 유형을 입력하세요 (예: INFP): ").strip().upper()

        if mbti == "종료":
            print("프로그램을 종료합니다.")
            break

        if mbti == "":
            print("MBTI 유형을 입력해주세요. (빈 값은 입력할 수 없습니다)")
            continue

        if mbti not in valid_mbti_types:
            print(f"'{mbti}'는 올바른 MBTI 유형이 아닙니다. "
                  "16가지 유형(예: INFP, ESTJ 등) 중 하나를 입력해주세요.")
            continue

        content_type = input("콘텐츠 유형을 입력하세요 (영화/책/음악): ").strip()

        if content_type == "종료":
            print("프로그램을 종료합니다.")
            break

        if content_type not in valid_content_types:
            print("콘텐츠 유형은 '영화', '책', '음악' 중 하나로 입력해주세요.")
            continue

        try:
            recommendations = get_recommendation(mbti, content_type)
            print(f"\n[{mbti} 유형을 위한 추천 {content_type} 목록]")
            for i, item in enumerate(recommendations, start=1):
                print(f"{i}. {item.strip()}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()