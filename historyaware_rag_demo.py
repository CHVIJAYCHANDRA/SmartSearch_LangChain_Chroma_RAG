from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SOURCE = os.getenv("SOURCE_FILE", "product-data.txt")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CONTEXTUALIZE_PROMPT = """Given the chat history and the latest user question, \
rewrite the question so it can be understood without the chat history.

Resolve pronouns and implicit references to the entities discussed earlier.
If the question is already standalone, return it unchanged.

Do NOT answer the question. Return only the rewritten question."""

QA_PROMPT = """You are an assistant answering questions about internal \
documentation.

Answer using only the context below. If the context does not contain the answer, \
say you do not know - do not guess, and do not use outside knowledge.

Keep the answer to three sentences at most.

Context:
{context}"""


# ----------------------------------------------------------------- indexing
def _fingerprint(path: Path) -> str:
    """Content hash + chunk params, so editing the corpus or changing chunking
    creates a new collection instead of querying a stale one."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{digest}"


def _load(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8").load()


@st.cache_resource(show_spinner="Building vector index...")
def get_store(source: str, api_key: str, persist_dir: str):
    """Cached across reruns AND sessions.

    This is the fix that matters: the corpus is embedded once per (content,
    chunking) combination rather than once per question.
    """
    path = Path(source)
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        api_key=api_key,
    )
    store = Chroma(
        collection_name=_fingerprint(path),
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    embedded = 0
    if store._collection.count() == 0:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(_load(path))
        store.add_documents(chunks)
        embedded = len(chunks)
    return store, embedded


@st.cache_resource(show_spinner=False)
def get_chain(source: str, api_key: str, persist_dir: str, model: str, k: int):
    """Build the two-stage chain: rewrite -> retrieve -> answer."""
    store, embedded = get_store(source, api_key, persist_dir)
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    history_aware_retriever = create_history_aware_retriever(
        llm,
        store.as_retriever(search_kwargs={"k": k}),
        ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        ),
    )
    qa_chain = create_stuff_documents_chain(
        llm,
        ChatPromptTemplate.from_messages(
            [
                ("system", QA_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        ),
    )
    return create_retrieval_chain(history_aware_retriever, qa_chain), embedded


# --------------------------------------------------------------------- app
st.set_page_config(page_title="SmartSearch - Conversational RAG", layout="wide")
st.title("SmartSearch")
st.caption(
    "Conversational document Q&A. Follow-up questions are rewritten into "
    "standalone queries before retrieval."
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add your key."
    )
    st.stop()

if not Path(SOURCE).exists():
    st.error(f"Source document not found: `{SOURCE}`")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    k = st.slider("Chunks retrieved (k)", 1, 8, 4)
    show_context = st.checkbox("Show retrieved chunks", value=False)
    show_rewrite = st.checkbox("Show rewritten query", value=True)
    st.divider()
    st.caption(f"Corpus: `{SOURCE}`")
    st.caption(f"Chunking: {CHUNK_SIZE} / {CHUNK_OVERLAP} overlap")

history = StreamlitChatMessageHistory(key="chat_history")

chain, embedded = get_chain(
    SOURCE, api_key, os.getenv("CHROMA_DIR", "./chroma_db"), model, k
)
if embedded:
    st.sidebar.success(f"Embedded {embedded} chunks (first build)")
else:
    st.sidebar.info("Index served from cache - no embedding cost")

with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        history.clear()
        st.rerun()

conversational_chain = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",  # key produced by create_retrieval_chain
)

for message in history.messages:
    st.chat_message(message.type).write(message.content)

if question := st.chat_input("Ask about the document..."):
    st.chat_message("human").write(question)

    with st.chat_message("ai"):
        turns_before = len(history.messages)
        started = time.perf_counter()
        with st.spinner("Retrieving..."):
            response = conversational_chain.invoke(
                {"input": question},
                {"configurable": {"session_id": "streamlit-session"}},
            )
        elapsed = time.perf_counter() - started

        st.write(response["answer"])

        docs = response.get("context", [])
        st.caption(f"{elapsed:.2f}s · {len(docs)} chunks · turn {turns_before // 2 + 1}")

        if show_rewrite and turns_before > 0:
            st.caption(
                "Follow-up detected: the question was rewritten into a "
                "standalone query before retrieval."
            )

        if show_context and docs:
            with st.expander(f"Retrieved chunks ({len(docs)})"):
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("page", doc.metadata.get("source", "?"))
                    st.markdown(f"**[{i}]** `{src}`")
                    st.info(" ".join(doc.page_content.split())[:400])
