import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import (create_history_aware_retriever, create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, 
                               MessagesPlaceholder, FewShotPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from fewshot import qa_examples
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
        index_name="pdf",
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
        ('system','''
         -사용자가 입력한 현재 질문과 이전 대화 기록(chat_history)을 바탕으로, 대화 맥락 없이도 완전히 이해할 수 있는 독립형 질문을 다시 작성하세요. 
         -사용자의 현재 질문이 불완전한 문장형태여도 이를 자연스럽게 보완하세요. 
         -질문에 대한 답변은 절대 작성하지 마세요.
         -필요할 경우에만 질문을 재구성하고, 그렇지 않으면 원문을 그대로 반환하세요.'''),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])

    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_retriever

# 퓨샷 프롬프트 생성================================================
def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("질문: {input}\n답변: {answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=qa_examples,  
        example_prompt=example_prompt, 
        prefix='다음 질문에 답변하세요::', 
        suffix="질문하세요: {input}",  
        input_variables=["input"],
    )

    formatted_few_shot_prompt = few_shot_prompt.format(input='{input}')   

    return formatted_few_shot_prompt

# 딕셔너리 json파일 로드함수 생성=====================================
def load_dict_file(path='keyword_dict.json'):
    with open(path, 'a', encoding='UTF-8') as file:
        return json.load(file)

# 딕셔너리 파일 str 변환 함수 생성=====================================
def build_dict_text(dictionary : dict) -> str:
    dict_text =  '\n'.join([
        f'{k} (", ".join(v["tags"]): {v["definition"]} [출처: {v["source"]}])' 
        for k, v in dictionary.items()
    ])
    return dict_text

# 질문 프롬프트 생성================================================
def build_qa_prompt():

    keyword_dict = load_dict_file()
    dictText = build_dict_text(keyword_dict)
    formatted_fs_prompt = build_few_shot_examples()

    qa_prompt = ChatPromptTemplate.from_messages([
        ('system',''' [identitiy]:당신은 청약 관련 전문가입니다. [context]와 [keyword_dictionary]를 참고하여 사용자의 질문에 답변하세요.
        - [context]: {context}
        - [keyword_dictionary]: {dictText}
        - 주어진 문서를 바탕으로 사용자의 청약 관련 질문에 정확하고 '자세하게' 답변해주세요.
        - 사용자의 질문 형태와 문법에는 관계 없이, 질문의 '내용'이 청약과 관계가 없을 경우엔 "청약 관련 질문만 답변할 수 있습니다"라고 답하세요.
        - 예를 들어, '청약사기는?' 의 형태로 불완전한 문장 형태여도 내용이 청약과 관계 있다면 최대한 문서를 찾아 답하세요.
        - 답변의 마지막에 해당 문서의 어떤 부분을 참조했는지 표시하세요.
        - "청약 관련 질문만 답변할 수 있습니다"를 답할 땐 참조 부분에 대한 답변은 넣지 마세요.'''),
        ('assistant', formatted_fs_prompt),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ]).partial(keyword_dict=dictText)
    
    return qa_prompt

# 전체 chain 구성================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    vectorDB = load_vectorstore()  #데이터 객체  

    DBretriever = vectorDB.as_retriever(search_kwargs={"k":3}) #데이터를검색기로 바꾼 객체

    history_retriever = build_history_retriever(
        llm=llm, 
        retriever=DBretriever)  # (llm+데이터검색기+대화히스토리와 현재질문넣어진질문프롬프트)=>히스토리검색기 객체  
    
    qa_prompt = build_qa_prompt() #질의응답 프롬프트 객체 

    qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt) # 질의응답프롬프트 + llm 연동 체인 생성 (Document 리스트를 문자열 변경 필요 없이 모델로 전송하게 하는 체인 생성. 매개변수이 prompt에는 꼭 "context"라는 입력 변수를 포함해야 함 이 컨텍스트는 내부 구조-아마도 create_retrieval_chain이 알아서 할당해준다.)

    retrieval_chain = create_retrieval_chain(history_retriever, qa_chain) #히스토리검색기 + 질의응답프롬프트 + llm 연동 체인 생성

    wrapped_chain = RunnableWithMessageHistory(
        retrieval_chain,
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