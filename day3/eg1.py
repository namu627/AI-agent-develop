from langchain.promptsimport PromptTemplate
from langchain_openaiimport ChatOpenAI
from langchain.output_parsersimport CommaSeparatedListOutputParser
from dotenvimport load_dotenv

load_dotenv()
# 1. Output Parser 설정
# → LLM 응답을 리스트로 변환하고
# 프롬프트에 형식 지침을 자동으로 추가
parser = CommaSeparatedListOutputParser()

# 2. PromptTemplate설정
# → {topic} 부분이 사용자 입력으로 채워짐
# → {format_instructions} 부분에 parser의 형식 지침이 자동 삽입
prompt = PromptTemplate(
    input_variables=["topic"],
    template="'{topic}'와 관련된 최신 키워드 5개를 쉼표로 구분해서 알려줘.\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 3. LLM 설정. 어떤모델 쓸건지
# → temperature: 0에 가까울수록 일관된 답변, 1에 가까울수록 창의적인 답변
llm= ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 4. Chain 연결
# → | 연산자로 prompt → llm→ parser 순서로 연결
# → 앞 단계의 출력이 자동으로 다음 단계의 입력으로 전달
chain = prompt | llm| parser

# 5. 실행
# → invoke()에 변수값만 넣으면 chain 전체가 순서대로 실행됨
result = chain.invoke({"topic": "인공지능"})
# result는 이미 리스트 형태로 변환된 상태
for keyword in result:
    print(f"-{keyword}")