import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
)

system_instruction = """당신은 10년 경력의 대기업 인사담당자이자 면접 코치입니다.
자소서와 채용공고를 분석해 실제 면접에서 나올 법한 질문을 생성하고
답변에 구체적이고 건설적인 피드백을 제공합니다.
- 예상 질문은 채용공고의 핵심 역량과 자소서 내용을 반영
- 답변 피드백은 STAR 기법 기준으로 평가
- 개선이 필요한 부분은 구체적인 개선 예시와 함께 제시
- 지원자가 자신감을 가질 수 있도록 강점도 함께 언급"""


@tool
def analyze_resume_and_job(resume: str, job_posting: str) -> str:
    """자기소개서와 채용공고를 함께 분석합니다.
    채용공고의 핵심 역량과 자소서의 경험을 매칭하여 강점, 부족한 점,
    면접에서 집중적으로 다뤄질 포인트를 정리합니다.

    Args:
        resume: 지원자의 자기소개서 전문
        job_posting: 채용공고 전문
    """
    prompt = f"""아래 자기소개서와 채용공고를 분석하세요.

[자기소개서]
{resume}

[채용공고]
{job_posting}

다음 형식으로 정리하세요.
1. 채용공고 핵심 역량 (3~5개)
2. 자소서에서 확인되는 강점 (역량과 매칭)
3. 근거가 부족하거나 보완이 필요한 부분
4. 면접에서 집중 검증될 포인트
"""
    return llm.invoke(prompt).content


@tool
def generate_interview_questions(analysis: str) -> str:
    """분석 결과를 바탕으로 예상 면접 질문 5가지를 생성합니다.

    Args:
        analysis: analyze_resume_and_job의 분석 결과 텍스트
    """
    prompt = f"""아래 분석 결과를 바탕으로 실제 면접에서 나올 법한 질문 5가지를 만드세요.

[분석 결과]
{analysis}

각 질문마다 다음을 포함하세요.
- 질문
- 질문 의도 (면접관이 무엇을 확인하려는가)
- 답변 시 반드시 포함해야 할 키워드
"""
    return llm.invoke(prompt).content


@tool
def feedback_answer(question: str, answer: str) -> str:
    """면접 질문에 대한 답변을 STAR 기법 기준으로 평가하고 피드백합니다.

    Args:
        question: 면접 질문
        answer: 지원자의 답변
    """
    prompt = f"""아래 면접 답변을 STAR 기법 기준으로 평가하세요.

[질문]
{question}

[답변]
{answer}

다음 형식으로 작성하세요.
1. STAR 항목별 평가 (Situation / Task / Action / Result 각각 충족 여부와 이유)
2. 잘한 점 (강점 구체적으로 언급)
3. 개선이 필요한 부분
4. 개선된 답변 예시 (실제 말할 수 있는 문장으로)
5. 총평 및 점수 (10점 만점)
"""
    return llm.invoke(prompt).content


tools = [analyze_resume_and_job, generate_interview_questions, feedback_answer]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)


def run_interview_coach(user_input: str) -> str:
    """외부에서 호출하는 진입점 함수"""
    try:
        result = agent_executor.invoke({"input": user_input})
        return result["output"]
    except Exception as e:
        return f"오류가 발생했습니다: {e}"


if __name__ == "__main__":
    print(run_interview_coach("면접 준비를 도와주세요."))