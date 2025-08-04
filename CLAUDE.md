# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요
LangChain과 Streamlit을 사용한 ChatPDF 애플리케이스. PDF 문서를 업로드하여 RAG(Retrieval-Augmented Generation) 시스템을 통해 질문응답을 할 수 있는 웹 애플리케이션입니다.

## 개발 명령어

### 애플리케이션 실행
```bash
streamlit run main.py
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

## 아키텍처

### 핵심 구성요소
- **Document Loading**: PyPDFLoader를 사용하여 PDF 파일 로드
- **Text Splitting**: RecursiveCharacterTextSplitter로 청크 단위 분할 (chunk_size=300, overlap=20)
- **Embedding**: OpenAI text-embedding-3-large 모델 사용 (직접 연결)
- **Vector Database**: ChromaDB를 사용한 벡터 저장소
- **Retrieval**: MultiQueryRetriever로 다중 쿼리 검색
- **LLM**: OpenRouter를 통한 DeepSeek Chat v3 모델 사용
- **UI**: Streamlit 웹 인터페이스

### 데이터 플로우
1. PDF 업로드 → 임시 파일 저장
2. 문서 로딩 및 텍스트 분할
3. 임베딩 생성 및 ChromaDB 저장
4. 사용자 질문 입력
5. MultiQueryRetriever로 관련 문서 검색
6. RAG 체인을 통한 답변 생성

## 환경 변수
- OpenRouter API 키가 필요합니다 (OPENROUTER_API_KEY)
- OpenAI API 키가 임베딩을 위해 필요합니다 (OPENAI_API_KEY)

## 주요 라이브러리 버전
- langchain: 0.3.20
- streamlit: 1.47.1
- chromadb: 0.6.3
- openai: 1.98.0

## 특이사항
- SQLite 호환성을 위해 pysqlite3를 사용하는 패치가 적용되어 있음 (main.py:1-3)
- ChromaDB 시스템 캐시 클리어 로직 포함 (main.py:55)