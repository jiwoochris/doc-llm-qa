import sys
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

load_dotenv()

class CLIHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        sys.stdout.write(token)
        sys.stdout.flush()


def main():
    # 기존 코드와 동일
    file_path = 

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    # PDF 파일 읽기
    reader = 
    
    # 데이터베이스 구축
    text_splitter = 
    all_splits = 

    vectorstore = 

    retriever = 

    while True:
        query_text = input("Enter your query (or 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break

        stream_handler = CLIHandler()

        # 응답 불러오기
        response = generate_response(retriever, query_text, stream_handler)
        print("\nAssistant:", response)

def generate_response(retriever, query_text, callback):
    llm = 

    rag_prompt = 

    rag_chain = 
    
    def log_and_invoke(query):
        docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(docs)} documents for query: {query}")
        print(docs)
        
        return rag_chain.invoke(query)

    return log_and_invoke(query_text)

if __name__ == "__main__":
    main()