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
    file_path = "data/(23.12.8. 정정) 2023000532 청계리버뷰자이 입주자모집공고문.pdf"

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    # PDF 파일 읽기
    reader = PdfReader(file_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += (text + " ")
    
    # 데이터베이스 구축
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )
    all_splits = text_splitter.create_documents([raw_text])
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))

    while True:
        query_text = input("Enter your query (or 'exit' to quit): ").strip()
        if query_text.lower() == 'exit':
            break

        stream_handler = CLIHandler()
        response = generate_response(retriever, query_text, stream_handler)
        print("\nAssistant:", response)

def generate_response(retriever, query_text, callback):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback])
    rag_prompt = PromptTemplate.from_template(
        "주어진 문서를 참고하여 사용자의 질문에 답변을 해줘.\n\n질문:{question}\n\n문서:{context}"
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    def log_and_invoke(query):
        docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(docs)} documents for query: {query}")
        print(docs)
        
        return rag_chain.invoke(query)

    return log_and_invoke(query_text)

if __name__ == "__main__":
    main()