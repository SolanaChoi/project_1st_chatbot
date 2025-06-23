import uuid
import streamlit as st
from llmGW import stream_answer

st.set_page_config(page_title='청약Q&A 챗봇', page_icon='🧞', layout="wide")

# Custom CSS 스타일링
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #e3f5ff 20%, #ffdbf8 80%);
    }
       
    /* 타이틀 스타일 */
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #ff9999, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        color: #500;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        padding: 0.5rem 1rem;
    }
    
    /* 사용 팁 카드 */
    .tips-card {
        background: linear-gradient(135deg, #ff9999 5%, #f5576c 95%);
        border-radius: 15px;
        padding: 1.0rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.5);
    }
    
    .tips-title {
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
    }
    
    .tips-list li {
        padding: 0.2rem 3rem;
        padding-left: 1.0rem;
        position: relative;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    .tips-list li:before {
        content: "✨";
        position: absolute;
        left: 0;
        top: 0.3rem;
    }      
</style>
""", unsafe_allow_html=True)

# 메인 컨테이너
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 타이틀 섹션
st.markdown("""
<div class="title-container">
    <h1 class="main-title">🧞청약지니🧞</h1>
    <div class="subtitle">
        이 챗봇은 국토교통부 공식문서인 ★ <b>2024 주택청약 FAQ.pdf</b> 파일을 기반으로 답변합니다.😄
    </div>
</div>
""", unsafe_allow_html=True)

# 사용 팁 카드
st.markdown("""
<div class="tips-card">
    <div class="tips-title">
        💡 사용 팁
    </div>
    <ul class="tips-list">
        <li><b>청약 관련</b> 궁금한 내용을 질문하세요.</li>
        <li><b>정확한 문장 형식</b>으로 입력할수록 답변이 좋아집니다.</li>
        <li>해당 문서에서 찾기 힘든 내용은 답변이 어렵습니다.</li>
        <li>문서 기반 답변임을 참고해 주세요.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

print('\n=====start=====')
print('BEFORE) st.session_state>>', st.session_state)

# url의 parameter에 session_id 저장 및 가져오기=======================
query_params = st.query_params

if 'session_id' in query_params:
    session_id = query_params.session_id
else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})

# Streamlit 내부 세션관리
if 'session_id' not in st.session_state:
    st.session_state.session_id = session_id

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print('AFTER) st.session_state>>', st.session_state)

# 이전 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 질문 입력 및 AI 답변 처리 영역 (페이지 하단에 별도 container로 분리)
placeholder='청약 관련 내용을 질문하세요.'  
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

    with st.spinner('답변을 생성 중입니다...😎'):
        answer = stream_answer(user_message=user_question, session_id=st.session_state.session_id)
        with st.chat_message('ai'):
            ai_message = st.write_stream(answer)
        st.session_state.message_list.append({'role':'ai','content':ai_message})

# 컨테이너 닫기
st.markdown('</div>', unsafe_allow_html=True)