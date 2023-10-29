from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data\주택임대차보호법(법률)(제19356호)(20230719).pdf")
pages = loader.load_and_split()

print(pages)