import streamlit as st

from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from dotenv import load_dotenv

load_dotenv()


def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        
        # loader
        raw_text = extract_text(uploaded_file)
        
        # splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 20,
            length_function = len,
            is_separator_regex = False,
        )
        all_splits = text_splitter.create_documents([raw_text])
        

        # storage
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        
        # retriever
        retriever = vectorstore.as_retriever()
        
        # generator
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
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

        response = log_and_invoke(query_text)

        return response


# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
# Query text
query_text = st.text_input('Enter your question:', placeholder = '질문을 입력하세요', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(response)