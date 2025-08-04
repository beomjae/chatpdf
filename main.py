__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], help="최대 5MB까지 업로드 가능합니다.")

# 파일 크기 체크 (5MB = 5 * 1024 * 1024 bytes)
if uploaded_file is not None and uploaded_file.size > 5 * 1024 * 1024:
  st.error("파일 크기가 5MB를 초과합니다. 더 작은 파일을 업로드해주세요.")
  uploaded_file = None
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

  # Text Splitter
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
  )

  texts = text_splitter.split_documents(pages)

  # Embedding
  embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
  )

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
      llm = ChatOpenAI(
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name="deepseek/deepseek-chat-v3-0324:free"
      )

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