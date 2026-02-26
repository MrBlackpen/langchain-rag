# LangChain-Rag

LangChain-Rag — a LangChain-built Retrieval-Augmented Generation (RAG) demo that ingests documents, constructs a vector store, and powers contextual conversational agents. It supports intra-chat (within a single conversation) and inter-chat (across multiple conversations) context, enables dynamic chat switching while preserving relevant history, and models explicit message roles: system messages (instructions/policy), human messages (user input), and AI messages (agent responses). Use this project to prototype context-aware assistants that require role-based messaging, session switching, and reliable retrieval from a vector knowledge base.

Features:
- Built with LangChain for modular RAG workflows.
- Intra-chat and inter-chat context support.
- Dynamic chat switching with history preservation.
- Explicit message roles: system, human, and AI messages.
- Contextual retrieval from a vector store.

Git:
- A `.gitignore` is included; initialize Git and commit to start tracking changes.

Prerequisites:
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

Quick start:
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Repository structure (high level):
- `app.py` — main entry point
- `vector_store.py`, `document_loader.py` — data & vector store
- `intent_classifier.py`, `knowledge_graph.py`, `memory_manager.py` — components

If you want, I can add usage examples or expand instructions.
