"""Ingest LangChain documentation into the Pinecone knowledge base.

Downloads the documentation pages, splits them into chunks, and upserts
them into the Pinecone index using deterministic chunk IDs, so re-running
the script replaces outdated chunks instead of duplicating them.

Run only when you want to refresh the knowledge base:

    uv run ingest.py
"""

import hashlib
import time

import requests
from langchain_core.documents import Document
from langchain_core.utils.uuid import uuid7
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

DOCS_BASE = "https://docs.langchain.com"
DOC_PATHS = [
    "oss/python/langchain/agents",
    "oss/python/deepagents/rag",
    "oss/python/langchain/tools",
    "oss/python/langchain/models",
    "oss/python/deepagents/retrieval",
    "oss/python/langchain/knowledge-base",
    "oss/python/langchain/middleware",
    "oss/python/deepagents/overview",
    "oss/python/deepagents/subagents",
    "oss/python/deepagents/streaming",
    "oss/python/deepagents/frontend/subagent-streaming",
    "oss/python/deepagents/backends",
    "oss/python/langgraph/overview",
    "oss/python/langgraph/quickstart",
]

# Shared configuration — agent.py imports these so both files stay in sync.
INDEX_NAME = "langchain-docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
    """Fetch LangChain documentation pages as Documents."""
    paths = doc_paths or DOC_PATHS
    docs: list[Document] = []
    for path in paths:
        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue
        source = f"{DOCS_BASE}/{path}"
        docs.append(
            Document(page_content=response.text, metadata={"source": source})
        )
    return docs


def chunk_id(source: str, text: str) -> str:
    """Deterministic ID for a chunk, so re-ingesting upserts instead of duplicating."""
    digest = hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()
    return f"{uuid7().hex[:8]}-{digest}"


def main() -> None:
    # Load documents
    docs = load_langchain_docs()
    print(f"Loaded {len(docs)} documentation pages.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split documentation into {len(all_splits)} chunks.")

    # Create embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create vector database
    pc = Pinecone()
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    else:
        print(f"Index {INDEX_NAME} already exists.")

    # Wait until the index is ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: {INDEX_NAME}")

    # Index the chunks (upsert: same content replaces its previous chunk)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    ids = [
        chunk_id(doc.metadata.get("source", "unknown"), doc.page_content)
        for doc in all_splits
    ]
    vector_store.add_documents(documents=all_splits, ids=ids)
    print(f"Indexed {len(all_splits)} chunks.")


if __name__ == "__main__":
    main()
