# SmartSearch_LangChain_Chroma_RAG

This repository contains robust Retrieval-Augmented Generation (RAG) applications built with LangChain, Chroma DB, and OpenAI. These tools are designed to read, embed, and query internal documents, simulating an enterprise-level internal search engine.

## Features

1. **Context-Aware PDF Q&A (`pdf_rag_demo.py`)**
   Uploads and parses a PDF document, chunks the text, stores the embeddings in Chroma, and allows a user to query the document with a strictly controlled "don't guess" prompt.

2. **Basic RAG (`rag_demo.py`)**
   A lightweight RAG implementation that reads from a local text file (like `product-data.txt`), splits the characters, and quickly builds a searchable vector database.

3. **History-Aware RAG with Streamlit (`historyaware_rag_demo.py`)**
   The most advanced application in this repository. It not only retrieves context from a local text file but also maintains a chat history using `StreamlitChatMessageHistory`. It uses a dedicated LLM chain to contextualize follow-up questions based on the chat history before querying the vector store.

## Technology Stack

- **LangChain**: Retrieval chains, history-aware retrievers, and prompt templates.
- **Chroma**: Fast, local vector database for storing document embeddings.
- **OpenAI Embeddings & GPT-4o**: For embedding chunks and generating accurate responses.
- **Streamlit**: For the interactive chat interface.

## How to Run

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Export your OpenAI API Key:
   ```bash
   export OPENAI_API_KEY="sk-your-api-key"
   ```

3. **Start an Application**:
   For example, run the History-Aware RAG application via Streamlit:
   ```bash
   streamlit run historyaware_rag_demo.py
   ```
