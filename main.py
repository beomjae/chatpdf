from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st


# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
  temp_dir = tempfile.TemporaryDirectory()
  temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
  with open(temp_file_path, "wb") as f:
    f.write(uploaded_file.getvalue())
  loader = PyPDFLoader(temp_file_path)
  pages = loader.load_and_split()
  return pages

# 업로드된 파일 처리
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)
  st.write(f"파일 업로드 완료: {uploaded_file.name}")
  st.write("---")

  # Text Splitter
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
  )

  texts = text_splitter.split_documents(pages)

  # Embedding
  embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

  import chromadb
  chromadb.api.client.SharedSystemClient.clear_system_cache()

  # Chroma DB
  db = Chroma.from_documents(texts, embeddings_model)

  # User Input
  st.header("PDF에게 질문해보세요!!")
  question = st.text_input("질문을 입력하세요")

  if st.button("질문하기"):
    with st.spinner("Wait for it..."):
      # Retriever
      llm = ChatOpenAI(temperature=0)

      retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
      )

      # Prompt Template
      prompt = hub.pull("rlm/rag-prompt")

      # Generate
      def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

      rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
      )

      # Question
      result = rag_chain.invoke({question})
      st.write(result)