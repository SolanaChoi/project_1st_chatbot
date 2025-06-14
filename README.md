# project_1st_chatbot

# 청약 QNA 챗봇 (RAG 기반)

이 리포지토리는 **RAG (Retrieval-Augmented Generation)** 기술을 활용하여 청약 관련 문서를 기반으로 질의응답(QNA) 챗봇을 구현한 프로젝트입니다.

## 주요 기능
- 사용자가 청약 관련 질문을 입력하면, RAG 모델이 관련 문서에서 정보를 검색해 답변을 생성합니다.
- Streamlit UI를 통해 대화형 인터페이스 제공
- 세션 별 대화 기록 관리 및 이전 대화 맥락을 반영한 질의 재구성 기능 포함

## 기술 스택
- Python, Streamlit
- LangChain 라이브러리 (ChatOpenAI, PineconeVectorStore 등)
- Pinecone 벡터 데이터베이스 (문서 임베딩 검색용)
- OpenAI GPT-4o 모델 (대화 생성 및 답변)

## 구조 설명
- `llmGW.py`: LangChain과 Pinecone 기반으로 RAG 체인 구성 및 답변 생성 로직 포함
- `chatbotGW.py`: Streamlit UI 구현 및 사용자 질문/답변 스트림 처리 담당

## 사용법
1. 필요한 API 키 및 환경변수를 `.env` 파일에 설정 (OpenAI, Pinecone 등)  
2. `pip install -r requirements.txt`로 필요한 라이브러리 설치  
3. `streamlit run chatbotGW.py` 실행 후 웹 UI에서 질문 입력 가능

## 주의사항
- 청약 관련 내용에 한해서 답변을 제공하며, 문서에 없는 정보는 답변하지 않습니다.
- 대화 내역은 세션별로 관리됩니다.

---

이 프로젝트는 청약 상담 업무에 AI를 적용해 효율성을 높이고자 개발되었습니다.
