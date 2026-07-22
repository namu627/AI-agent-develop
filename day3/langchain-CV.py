import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)

# 코드와 같은 폴더의 자소서.txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_FILE = os.path.join(BASE_DIR, "자소서.txt")

system_instruction = """당신은 10년 경력의 취업 컨설턴트입니다.
자소서를 분석하고 채용공고에 맞게 개선하는 것이 전문입니다.
- 자소서의 강점과 개선점을 구체적으로 피드백
- 채용공고의 핵심 역량과 키워드를 추출
- 추출한 키워드를 반영해 개선된 자소서 생성"""


# ===== 파일 입출력 =====
def read_txt(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "파일 인코딩을 인식할 수 없습니다.")


def write_txt(path: str, content: str) -> None:
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        bak = f"{base}_backup_{stamp}{ext}"
        with open(bak, "w", encoding="utf-8") as f:
            f.write(read_txt(path))
        print(f"[백업 생성] {os.path.basename(bak)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ===== Tools =====
@tool
def load_resume_file(dummy: str = "") -> str:
    """코드와 같은 폴더에 있는 자소서.txt 파일을 읽어 자기소개서 원문을 반환한다."""
    try:
        return read_txt(RESUME_FILE)
    except FileNotFoundError:
        return "자소서.txt 파일이 없습니다."
    except Exception as e:
        return f"파일 읽기 실패: {e}"


@tool
def save_resume_file(content: str) -> str:
    """개선된 자기소개서 내용을 같은 폴더의 자소서.txt에 덮어써서 저장한다. 기존 파일은 자동 백업된다."""
    try:
        write_txt(RESUME_FILE, content)
        return f"저장 완료: {RESUME_FILE}"
    except Exception as e:
        return f"파일 저장 실패: {e}"


@tool
def review_resume(resume: str) -> str:
    """자소서 텍스트를 입력받아 강점과 개선점을 구체적으로 피드백한다."""
    prompt = f"""다음 자소서를 분석해 주세요.

[자소서]
{resume}

아래 형식으로 작성하세요.
1. 강점 (3가지 이상, 근거 문장 인용)
2. 개선점 (3가지 이상, 구체적 수정 방향 제시)
3. 총평 (5줄 이내)"""
    return llm.invoke(prompt).content


@tool
def analyze_job_posting(job_posting: str) -> str:
    """채용공고 텍스트를 입력받아 핵심 역량과 키워드를 추출한다."""
    prompt = f"""다음 채용공고를 분석해 주세요.

[채용공고]
{job_posting}

아래 형식으로 작성하세요.
1. 직무 요약
2. 핵심 역량 (우선순위 순 5가지)
3. 필수 키워드 (자소서에 반드시 들어가야 할 단어 10개)
4. 우대 사항 요약"""
    return llm.invoke(prompt).content


@tool
def improve_resume(resume: str, job_posting: str) -> str:
    """자소서와 채용공고를 입력받아 채용공고 맞춤 개선 버전 자소서를 생성한다."""
    prompt = f"""아래 자소서를 채용공고에 맞게 개선해 주세요.

[자소서]
{resume}

[채용공고]
{job_posting}

요구사항:
- 채용공고의 핵심 키워드를 자연스럽게 반영
- 성과는 수치 기반으로 구체화
- 설명이나 머리말 없이 '개선된 자소서 본문'만 출력
- 본문이 끝나면 '---' 구분선 뒤에 '주요 수정 포인트'를 3~5줄로 정리"""
    return llm.invoke(prompt).content


@tool
def generate_interview_questions(resume: str, job_posting: str) -> str:
    """자소서와 채용공고를 바탕으로 면접 예상 질문과 답변 가이드를 생성한다."""
    prompt = f"""다음 자소서와 채용공고를 바탕으로 면접 예상 질문을 만들어 주세요.

[자소서]
{resume}

[채용공고]
{job_posting}

아래 형식으로 작성하세요.
1. 자소서 기반 질문 5개 (+ 답변 방향 힌트)
2. 직무/기술 역량 질문 5개 (+ 답변 방향 힌트)
3. 압박 질문 3개 (+ 대응 전략)"""
    return llm.invoke(prompt).content


tools = [
    load_resume_file,
    save_resume_file,
    review_resume,
    analyze_job_posting,
    improve_resume,
    generate_interview_questions,
]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)


# ===== 입력 헬퍼 =====
def multiline_input(msg: str) -> str:
    print(msg)
    print("(입력 완료 후 빈 줄에서 Enter)")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def print_menu(state):
    r = "불러옴" if state["resume"] else "미입력"
    j = "입력됨" if state["job_posting"] else "미입력"
    print("\n" + "=" * 55)
    print("  자소서 리뷰 & 채용공고 맞춤 개선 에이전트")
    print(f"  [자소서: {r}] [채용공고: {j}]")
    print("=" * 55)
    print("1) 자소서 불러오기 (자소서.txt)")
    print("2) 채용공고 입력")
    print("3) 리뷰 시작 (분석 + 개선)")
    print("4) 면접 예상 질문 생성")
    print("5) 종료")
    print("=" * 55)


def main():
    state = {"resume": "", "job_posting": "", "improved": ""}
    print(f"[작업 폴더] {BASE_DIR}")
    print(f"[자소서 파일] {os.path.basename(RESUME_FILE)}")

    while True:
        print_menu(state)
        choice = input("메뉴 선택 >> ").strip()

        # 1) 자소서.txt 불러오기
        if choice == "1":
            if not os.path.exists(RESUME_FILE):
                print(f"'{os.path.basename(RESUME_FILE)}' 파일이 없습니다. 코드와 같은 폴더에 만들어 주세요.")
                continue
            try:
                text = read_txt(RESUME_FILE)
            except Exception as e:
                print(f"읽기 실패: {e}")
                continue
            if not text.strip():
                print("파일이 비어 있습니다.")
                continue
            state["resume"] = text
            memory.chat_memory.add_user_message(f"[자소서 로드]\n{text}")
            memory.chat_memory.add_ai_message("자소서를 불러왔습니다.")
            print(f"\n불러오기 완료 ({len(text)}자)")
            print("-" * 45)
            print(text[:300] + ("..." if len(text) > 300 else ""))
            print("-" * 45)

        # 2) 채용공고 입력
        elif choice == "2":
            text = multiline_input("\n[채용공고를 입력하세요]")
            if not text.strip():
                print("입력된 내용이 없습니다.")
                continue
            state["job_posting"] = text
            memory.chat_memory.add_user_message(f"[채용공고 등록]\n{text}")
            memory.chat_memory.add_ai_message("채용공고를 저장했습니다.")
            print("채용공고가 저장되었습니다.")

        # 3) 리뷰 + 저장 확인
        elif choice == "3":
            if not state["resume"] or not state["job_posting"]:
                print("자소서 불러오기(1)와 채용공고 입력(2)을 먼저 해 주세요.")
                continue

            query = f"""아래 자소서와 채용공고를 분석해 주세요.
review_resume, analyze_job_posting, improve_resume 도구를 모두 사용하고
결과를 [자소서 리뷰] / [채용공고 분석] / [개선된 자소서] 순서로 정리해 주세요.

[자소서]
{state['resume']}

[채용공고]
{state['job_posting']}"""
            result = agent_executor.invoke({"input": query})
            print("\n===== 분석 결과 =====\n")
            print(result["output"])

            # 저장용 개선본 확보 (본문만 분리)
            improved_raw = improve_resume.invoke({
                "resume": state["resume"],
                "job_posting": state["job_posting"],
            })
            state["improved"] = improved_raw.split("---")[0].strip()

            # 저장 여부 확인
            print("\n" + "=" * 45)
            print("개선된 자소서를 '자소서.txt'에 저장할까요?")
            print("(기존 파일은 자동으로 백업됩니다)")
            print("=" * 45)
            print("\n--- 저장될 내용 미리보기 ---")
            print(state["improved"][:400] + ("..." if len(state["improved"]) > 400 else ""))
            print("-" * 45)

            ans = input("저장하시겠습니까? (y/n) >> ").strip().lower()
            if ans in ("y", "yes", "ㅇ", "예"):
                try:
                    write_txt(RESUME_FILE, state["improved"])
                    state["resume"] = state["improved"]
                    print(f"저장 완료: {RESUME_FILE}")
                except Exception as e:
                    print(f"저장 실패: {e}")
            else:
                print("저장하지 않았습니다.")

        # 4) 면접 질문
        elif choice == "4":
            if not state["resume"] or not state["job_posting"]:
                print("자소서 불러오기(1)와 채용공고 입력(2)을 먼저 해 주세요.")
                continue
            query = f"""generate_interview_questions 도구를 사용해 면접 예상 질문을 만들어 주세요.

[자소서]
{state['resume']}

[채용공고]
{state['job_posting']}"""
            result = agent_executor.invoke({"input": query})
            print("\n===== 면접 예상 질문 =====\n")
            print(result["output"])

        elif choice == "5" or choice == "종료":
            print("프로그램을 종료합니다.")
            break

        else:
            if choice:
                result = agent_executor.invoke({"input": choice})
                print("\n" + result["output"])


if __name__ == "__main__":
    main()