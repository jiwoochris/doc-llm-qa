import streamlit as st

from docllmqa.loader.pdf_loader import PDFMinerLoader
from docllmqa.splitter.text_splitter import CharacterTextSplitter
from docllmqa.storage.openai import ChromaVectorStore
from docllmqa.generator.openai import LLM



def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        
        # loader
        loader = PDFMinerLoader()
        data = loader.load(uploaded_file)
        
        # splitter
        text_splitter = CharacterTextSplitter()
        all_splits = text_splitter.split_documents(data)
        
        # storage
        storage = ChromaVectorStore()
        vector_store = storage.documents_embedding(all_splits)
        
        # retriever
        retriever = vector_store.as_retriever()
        
        # generator
        generator = LLM(retriever)
        response = generator.generate_response(query_text)
        
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