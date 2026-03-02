# LangChain-Rag/memory_manager.py

from vector_store import search_similar
from knowledge_graph import query_kg
import spacy

# Load spaCy once (very important)
nlp = spacy.load("en_core_web_sm")


def extract_entity(query):
    """
    Extract main named entity using spaCy instead of LLM.
    Prioritizes ORG, GPE, PERSON, PRODUCT, EVENT.
    """

    doc = nlp(query)

    # Prefer important entity types
    priority_labels = ["ORG", "GPE", "PERSON", "PRODUCT", "EVENT"]

    for label in priority_labels:
        for ent in doc.ents:
            if ent.label_ == label:
                return ent.text.strip()

    # Fallback: longest noun phrase
    noun_chunks = list(doc.noun_chunks)
    if noun_chunks:
        return max(noun_chunks, key=lambda x: len(x.text)).text.strip()

    return query  # last fallback


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