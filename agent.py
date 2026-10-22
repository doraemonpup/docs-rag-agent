import requests
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Load documents
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

docs = load_langchain_docs()
print(f"Loaded {len(docs)} documentation pages.")

# total_chars = sum(len(doc.page_content) for doc in docs)
# print(f"Total characters: {total_chars}")
# print(docs[0].page_content[:500])

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"Split documentation into {len(all_splits)} chunks.")

# Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Create vector database
index_name = "vector-langchain"
pc = Pinecone()
pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    )
)

index = pc.Index(index_name)

# Create vector store
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index
)

# Index the chunks
vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(all_splits)} chunks.")