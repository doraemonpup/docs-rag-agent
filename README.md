# Agentic RAG for LangChain Documents

## Tech Stacks
- LLM from OpenRouter
- DeepAgents
- LangChain
- HuggingFaceEmbeddings
- Pinecone Vector Database

## Project Structure
- `ingest.py` — Downloads LangChain docs, splits them into chunks, and upserts them into Pinecone. Run only when you want to build or refresh the knowledge base.
- `agent.py` — Connects to the existing Pinecone index and runs the Deep Agent. No downloading or re-indexing on normal runs.
- `prompts.py` — All agent instructions and prompt configuration.

## Setup
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Configure your API keys in a `.env` file:
   ```
   OPENROUTER_API_KEY=...
   PINECONE_API_KEY=...
   ```

## Usage
1. **Build the knowledge base** (first time, or whenever the docs should be refreshed):
   ```bash
   uv run ingest.py
   ```
   Chunk IDs are derived from chunk content, so re-running this script
   updates existing chunks instead of creating duplicates.

2. **Ask questions**:
   ```bash
   uv run agent.py
   ```
