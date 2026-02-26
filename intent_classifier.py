# LangChain-Rag/intent_classifier.py
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL)

def classify_intent(query):
    factual_keywords = [
        "how many",
        "when",
        "where",
        "who",
        "count",
        "number",
        "year",
        "population",
        "total"
    ]

    query_lower = query.lower()

    if any(keyword in query_lower for keyword in factual_keywords):
        return "factual"

    return "conversational"