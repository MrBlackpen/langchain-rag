# Langchain-Rag/memory_manager.py
from vector_store import search_similar
from knowledge_graph import query_kg, extract_entity
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL)


def extract_entity(query):
    prompt = f"""
    Extract the main entity name from this question.

    Question: {query}

    Return only the entity name.
    """

    response = llm.invoke(prompt)
    return response.content.strip()


def retrieve_context(query, intent, workspace_id):

    vector_context = search_similar(query, workspace_id)

    if intent == "factual":
        entity = extract_entity(query)
        kg_data = query_kg(entity, workspace_id)

        return f"""
        Knowledge Graph Data:
        {kg_data}

        Vector Memory:
        {vector_context}
        """

    return vector_context