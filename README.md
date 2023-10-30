# doc-llm-qa

# 🦜🔗 Ask the Doc 앱

**Ask the Doc 앱**은 사용자가 업로드한 PDF 문서를 바탕으로 질문에 대한 답변을 생성해주는 앱입니다.

<img src="asset/demo.png" alt="Ask the Doc 앱 데모" width="50%">

## 주요 특징

1. PDF 문서 내용 추출: `pdfminer`를 사용하여 PDF 문서의 텍스트를 추출합니다.
2. 텍스트 분할: `CharacterTextSplitter`를 사용하여 텍스트를 적절한 크기의 문서로 분할합니다.
3. 임베딩 및 저장: `Chroma` 및 `OpenAIEmbeddings`를 활용하여 텍스트 임베딩을 생성하고 저장합니다.
4. 문서 검색: 질문에 관련된 문서를 검색하여 결과를 반환합니다.
5. 답변 생성: GPT-4 모델을 활용하여 질문에 대한 답변을 생성합니다.

## 사용법

1. 앱을 시작하면 `🦜🔗 Ask the Doc App` 제목이 표시됩니다.
2. `Upload an article` 버튼을 사용하여 PDF 문서를 업로드합니다.
3. 업로드한 문서를 바탕으로 질문을 `Enter your question:` 입력창에 입력합니다. (예시: `질문을 입력하세요`)
4. `Submit` 버튼을 클릭하여 질문에 대한 답변을 생성합니다.
5. 생성된 답변은 페이지 하단에 표시됩니다.

