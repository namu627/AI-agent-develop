from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# 1. 채용공고 문서 로드
# → job_posting.txt 파일에 채용공고 내용을 미리 저장해두고 불러옴
# → encoding="utf-8": 한국어 텍스트가 포함된 경우 반드시 지정
loader = TextLoader("job_posting.txt", encoding="utf-8")
documents = loader.load()# 파일 전체 내용이 하나의 Document 객체로 로드됨

# 2. 문서를 chunk로 분할
# → LLM은 한 번에 처리할 수 있는 텍스트 길이(컨텍스트 윈도우)가 제한됨
# → 문서 전체를 한 번에 넣는 대신 작은 단위로 잘라서 관리
# → chunk_size=500: 한 chunk의 최대 글자 수
# → chunk_overlap=50: 앞뒤 chunk와 50글자씩 겹치게 해서 문맥이 끊기지 않도록 함
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
# → chunks: 문서가 여러 개의 작은 조각으로 분할된 리스트

# 3. 임베딩 모델로 벡터 변환 후 FAISS에 저장
# → 임베딩: 텍스트를 숫자 배열(벡터)로 변환하는 과정
# → 의미가 유사한 텍스트는 벡터 공간에서 가까운 위치에 놓임
# → FAISS: Facebook에서 개발한 벡터 검색 라이브러리. 로컬에서 무료로 사용 가능
embeddings = OpenAIEmbeddings()
vectorstore= FAISS.from_documents(chunks, embeddings)
# → vectorstore: 모든 chunk의 벡터가 저장된 데이터베이스

# 4. 검색기 생성
# → k=3: 사용자 질문과 가장 유사한 chunk를 3개 검색
# → k값이 클수록 더 많은 문맥을 LLM에 전달하지만 토큰 소모도 증가
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})