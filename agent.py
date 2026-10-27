"""Connect to the existing Pinecone index and run the Deep Agent.

No downloading or re-indexing happens here — run ingest.py first to
build the knowledge base.
"""

from deepagents.backends import StateBackend
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

from ingest import EMBEDDING_MODEL_NAME, INDEX_NAME
from prompts import CHUNK_ANALYST_INSTRUCTIONS, INSTRUCTIONS

load_dotenv()

# Connect to the existing vector store
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

pc = Pinecone()
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index
)

backend = StateBackend()

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = vector_store.similarity_search(query, k=4)
    batch_id = uuid7().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode('utf-8')))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )

# Create the agents
chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved documentation chunk file. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

# OpenRouter's model || OpenAI's model
model = init_chat_model(
    model='openrouter:openrouter/free', # openai:gpt-5-nano
)

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent]
)

# Run the RAG agent
EXAMPLE_QUERY = "What are LangChain, LangGraph, and DeepAgents, and how do they differ in terms of purpose, architecture, and capabilities? When should I use each one, and how do I choose the right tool for my specific use case?"

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [HumanMessage(content=EXAMPLE_QUERY)]}
    )

    for msg in result.get("messages", []):
        if msg.text:
            print(msg.text)
