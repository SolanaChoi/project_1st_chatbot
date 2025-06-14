# setup_vectorstore.py - 처음 한 번만 실행하는 스크립트

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

def pdf_to_doc(pdf_path: str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents


# PDF 문서 로드
pdf_path = "/Users/apple/Desktop/Programming/chatbot/2024subscription_FAQ.pdf"
documents = pdf_to_doc(pdf_path)

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
    chunk_overlap=50)
doc_list = text_splitter.split_documents(documents)
print(f"분할된 청크 수: {len(doc_list)}")

# 임베딩 생성
print("임베딩 생성 중...")
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

# Pinecone 벡터스토어 생성 + 테스트 처음 10개만
print("Pinecone에 저장 중...")

batch_size = 50  # 한 번에 50개씩만 처리
vectorstore = None

for i in range(0, len(doc_list), batch_size):
    batch = doc_list[i:i+batch_size]
    print(f"배치 {i//batch_size + 1} 처리 중... ({len(batch)}개)")
    
    if vectorstore is None:
        # 첫 번째 배치로 vectorstore 생성
        vectorstore = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name='chat'
        )
    else:
        # 나머지 배치들 추가
        vectorstore.add_documents(batch)

