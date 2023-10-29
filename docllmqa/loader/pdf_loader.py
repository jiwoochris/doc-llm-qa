from pdfminer.high_level import extract_text

class PDFMinerLoader:
    def __init__(self):
        pass

    def load(self, file_path):
        return extract_text(file_path)

# # Usage:
# loader = PDFMinerLoader("data\주택임대차보호법(법률)(제19356호)(20230719).pdf")
# data = loader.load()
# print(data)
