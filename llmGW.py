import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import (create_history_aware_retriever, create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# llm 생성
def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

llm = load_llm()

# Embedding+ 벡터스토어index 가져오기===============================
def load_vectorstore():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="chat",
        embedding=OpenAIEmbeddings(model='text-embedding-3-large')
    )
    return vectorstore

# 세션 히스토리 저장 기능 설정========================================
store = {}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 히스토리 기반 리트리버
def build_history_retriever(llm, retriever):
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ('system','''이전 대화 기록과 최신 사용자 질문을 바탕으로, 대화 맥락 없이도 완전히 이해할 수 있는 독립형 질문을 다시 작성하세요. 질문에 대한 답변은 절대 작성하지 말고, 필요할 경우에만 질문을 재구성하고, 그렇지 않으면 원문을 그대로 반환하세요.'''),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])
    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_retriever

# 질문 프롬프트 생성================================================
def build_qa_prompt():
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system','''당신은 청약 관련 전문가입니다. [context]를 참고하여 사용자의 질문에 답변하세요. [context]: {context}
        - 주어진 문서를 바탕으로 사용자의 청약 관련 질문에 정확하고 '자세하게' 답변해주세요.
        - 문서에 없는 정보는 "청약 관련 질문만 답변할 수 있습니다"라고 답하세요.
        - 답변의 마지막에 해당 문서의 어떤 부분을 참조했는지 표시하세요.'''),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])
    return qa_prompt

# 전체 chain 구성================================================
def build_conversational_chain():
    vectorDB = load_vectorstore()
    DBretriever = vectorDB.as_retriever(search_kwargs={"k":3})

    history_retriever = build_history_retriever(llm=llm, retriever=DBretriever)
    
    qa_prompt = build_qa_prompt()
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    rag_chain = create_retrieval_chain(history_retriever, qa_chain)

    wrapped_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    ).pick('answer')

    return wrapped_chain
    
# AI Message==================================================
def stream_answer(user_message, session_id='default'):
    wrapped_chain = build_conversational_chain()

    answer = wrapped_chain.stream(
        {'input': user_message},
        {'configurable':{'session_id':session_id}},
    )

    print(f'대화이력>>>{get_session_history(session_id)} \n\n')
    print(f'[stream_answer 함수 내 출력]session_id >>>{session_id}\n')

    return answer