from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub

from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self, retriever):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        
        self.retriever = retriever
        
    def generate_response(self, question):
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | self.rag_prompt
            | self.llm 
        )
        
        # response = rag_chain.invoke(question)
        
        # Define a wrapper function to log retrieval results
        def log_and_invoke(query):
            docs = self.retriever.get_relevant_documents(query)
            print(f"Retrieved {len(docs)} documents for query: {query}")
            print(docs)
            
            return rag_chain.invoke(query)

        response = log_and_invoke(question)
        
        response = response.content
        
        return response