# Streamlit기본 예시
import streamlit as st

st.title("면접 준비 에이전트")
user_input= st.text_input("질문을 입력하세요")
if st.button("전송"):
    st.write(f"입력한 내용: {user_input}")

# 제목과 설명
st.title("면접 준비 에이전트")
st.markdown("자소서와 채용공고를 입력하고 면접 준비를 시작하세요!")
# 긴 텍스트 입력
resume = st.text_area("자소서를 입력하세요", height=200)
job_posting= st.text_area("채용공고를 입력하세요", height=200)
# 버튼
if st.button("정보 저장"):
    st.success("저장되었습니다!")
# 사이드바
with st.sidebar:
    st.header("정보 입력")

# 채팅 메시지
with st.chat_message("user"):
    st.write("면접 질문 생성해줘")
with st.chat_message("assistant"):
    st.write("예상 면접 질문을 생성했습니다...")
# 채팅 입력
user_input= st.chat_input("질문을 입력하세요")
# 스피너
with st.spinner("AI가 분석 중입니다..."):
    pass
# session_state: 대화 기록 유지
if "messages" not in st.session_state:
    st.session_state.messages= []