import streamlit as st
from agent import run_interview_coach

st.set_page_config(page_title="AI 면접 준비 코치", page_icon="🎯", layout="wide")
st.title("🎯 AI 면접 준비 코치")
st.caption("자소서와 채용공고를 입력하면 예상 질문 생성과 STAR 기반 답변 피드백을 제공합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_posting" not in st.session_state:
    st.session_state.job_posting = ""

with st.sidebar:
    st.header("📄 내 정보 입력")

    resume_input = st.text_area(
        "자기소개서",
        value=st.session_state.resume,
        height=250,
        placeholder="자기소개서 전문을 붙여넣으세요.",
    )
    job_input = st.text_area(
        "채용공고",
        value=st.session_state.job_posting,
        height=250,
        placeholder="지원하려는 채용공고를 붙여넣으세요.",
    )

    if st.button("💾 정보 저장", use_container_width=True):
        st.session_state.resume = resume_input
        st.session_state.job_posting = job_input
        if resume_input.strip() and job_input.strip():
            st.success("정보가 저장되었습니다!")
        else:
            st.warning("자소서와 채용공고를 모두 입력해주세요.")

    st.divider()
    st.subheader("저장 상태")
    st.write("자기소개서:", "✅ 저장됨" if st.session_state.resume else "❌ 미입력")
    st.write("채용공고:", "✅ 저장됨" if st.session_state.job_posting else "❌ 미입력")

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("💡 사용 예시")
    st.markdown(
        "- 내 자소서와 채용공고를 분석해줘\n"
        "- 예상 면접 질문 5개 만들어줘\n"
        "- 방금 질문에 대한 내 답변 피드백해줘"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("무엇을 도와드릴까요?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    if st.session_state.resume and st.session_state.job_posting:
        context = (
            f"[저장된 자기소개서]\n{st.session_state.resume}\n\n"
            f"[저장된 채용공고]\n{st.session_state.job_posting}\n\n"
        )

    full_input = context + f"[사용자 요청]\n{user_input}"

    with st.chat_message("assistant"):
        with st.spinner("면접 코치가 답변을 작성 중입니다..."):
            response = run_interview_coach(full_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})