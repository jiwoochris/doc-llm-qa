from langchain.text_splitter import RecursiveCharacterTextSplitter

class CharacterTextSplitter:    
    def __init__(self, chunk_size=500, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap  = self.chunk_overlap,
            length_function = len,
            is_separator_regex = False,
        )
        
    def split_documents(self, text):
        all_splits = self.text_splitter.create_documents([text])
        
        return all_splits