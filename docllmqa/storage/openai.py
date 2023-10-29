from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

class ChromaVectorStore:    
    def __init__(self):
        pass
        
    def documents_embedding(self, splits):
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        
        return vectorstore