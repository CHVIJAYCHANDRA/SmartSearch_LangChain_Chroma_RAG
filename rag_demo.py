from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

QA_SYSTEM_PROMPT = """You are an assistant answering questions about internal \
documentation.

Answer using only the context below. If the context does not contain the answer, \
say you do not know - do not guess, and do not use outside knowledge.

Keep the answer to three sentences at most.

Context:
{context}"""


def require_api_key() -> str:
    """Fail early with an actionable message rather than deep inside the SDK."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit(
            "OPENAI_API_KEY is not set.\n"
            "  cp .env.example .env   then add your key, or\n"
            "  export OPENAI_API_KEY=sk-..."
        )
    return key


def load_documents(path: Path):
    """Dispatch on file extension. This is the only difference between what used
    to be two near-identical scripts."""
    if not path.exists():
        sys.exit(f"Source file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    sys.exit(f"Unsupported file type '{suffix}'. Use .txt, .md or .pdf")


def fingerprint(path: Path) -> str:
    """Content hash + chunking params.

    Included in the collection name so changing chunk_size or the source file
    produces a new collection instead of silently querying a stale index.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{digest}"


def get_vector_store(path: Path, embeddings, persist_dir: str, rebuild: bool):
    """Load the persisted index if it matches the source, else build it.

    Returns (store, was_built) so the caller can report which path was taken.
    """
    collection = fingerprint(path)
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    if rebuild or store._collection.count() == 0:
        if rebuild:
            store.delete_collection()
            store = Chroma(
                collection_name=collection,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
        docs = load_documents(path)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ).split_documents(docs)
        if not chunks:
            sys.exit(f"No text extracted from {path}")
        store.add_documents(chunks)
        return store, len(chunks)

    return store, 0


def build_chain(store, model: str, api_key: str, k: int):
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", QA_SYSTEM_PROMPT), ("human", "{input}")]
    )
    retriever = store.as_retriever(search_kwargs={"k": k})
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


def answer(chain, question: str, show_context: bool) -> None:
    started = time.perf_counter()
    response = chain.invoke({"input": question})
    elapsed = time.perf_counter() - started

    print(f"\n{response['answer']}")
    docs = response.get("context", [])
    print(f"\n[{elapsed:.2f}s · {len(docs)} chunks retrieved]")

    if show_context:
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("page", doc.metadata.get("source", "?"))
            snippet = " ".join(doc.page_content.split())[:220]
            print(f"  [{i}] ({source}) {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Q&A over a local document")
    parser.add_argument("--source", default="product-data.txt", help=".txt, .md or .pdf")
    parser.add_argument("--question", help="ask once and exit (non-interactive)")
    parser.add_argument("--k", type=int, default=RETRIEVAL_K, help="chunks to retrieve")
    parser.add_argument("--rebuild", action="store_true", help="force reindex")
    parser.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    args = parser.parse_args()

    api_key = require_api_key()
    path = Path(args.source)
    persist_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        api_key=api_key,
    )

    started = time.perf_counter()
    store, chunks_added = get_vector_store(path, embeddings, persist_dir, args.rebuild)
    setup = time.perf_counter() - started

    if chunks_added:
        print(f"Indexed {path.name}: {chunks_added} chunks embedded in {setup:.2f}s")
    else:
        print(f"Loaded cached index for {path.name} in {setup:.2f}s (no embedding cost)")

    chain = build_chain(store, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"), api_key, args.k)

    if args.question:
        answer(chain, args.question, args.show_context)
        return

    print("Ask questions about the document. Blank line or Ctrl-C to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        answer(chain, question, args.show_context)


if __name__ == "__main__":
    main()
