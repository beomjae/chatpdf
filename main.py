# 로컬 개발시에는 주석처리
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
from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st
import chromadb
from streamlit_extras.buy_me_a_coffee import button


# 제목
st.title("ChatPDF")
st.write("---")

# OPENAI 키 입력받기
openai_key = st.text_input('OPENAI_API_KEY', type='password')

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], help="최대 5MB까지 업로드 가능합니다.")

# 파일 크기 체크 (5MB = 5 * 1024 * 1024 bytes)
if uploaded_file is not None and uploaded_file.size > 5 * 1024 * 1024:
  st.error("파일 크기가 5MB를 초과합니다. 더 작은 파일을 업로드해주세요.")
  uploaded_file = None
st.write("---")

# Buy me a coffee
button(username="jejupeter", floating=True, width=221)

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
    openai_api_key=openai_key
  )

  chromadb.api.client.SharedSystemClient.clear_system_cache()

  # Chroma DB
  db = Chroma.from_documents(texts, embeddings_model)

  # 스트밍 처리할 Handler 생성
  class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
      self.container = container
      self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
      self.text += token
      self.container.markdown(self.text)


  # User Input
  st.header("PDF에게 질문해보세요!!")
  question = st.text_input("질문을 입력하세요")

  if st.button("질문하기"):
    with st.spinner("Wait for it..."):
      # Retriever
      llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_key,
      )

      retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
      )

      # Prompt Template
      prompt = hub.pull("rlm/rag-prompt")

      # Generate
      chat_box = st.empty()
      stream_handler = StreamHandler(chat_box)
      generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
      
      def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

      rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
      )
      # Question
      result = rag_chain.invoke(question)