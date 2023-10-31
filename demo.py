import streamlit as st

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

from dotenv import load_dotenv

load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def generate_response(uploaded_file, query_text, callback):
    # Load document if file is uploaded
    if uploaded_file is not None:
        
        # loader
        raw_text =
        
        # splitter
        text_splitter = 
        all_splits = 
        

        # storage
        vectorstore = 
        
        # retriever
        retriever = 
        
        # generator
        llm = 
        
        rag_prompt = 
        
        # Chaining
        rag_chain = 
        
        def log_and_invoke(query):
            docs = retriever.get_relevant_documents(query)
            print(f"Retrieved {len(docs)} documents for query: {query}")
            print(docs)
            
            return rag_chain.invoke(query)

        response = log_and_invoke(query_text)

        return response


# Page title
st.set_page_config(page_title='🦜🔗 문서 기반 질문 답변 챗봇')
st.title('🦜🔗 문서 기반 질문 답변 챗봇')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content='안녕하세요! 저는 문서를 기반으로 답변해주는 챗봇입니다. 어떤게 궁금하신가요?'
        )
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        response = generate_response(uploaded_file, prompt, stream_handler)
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        )