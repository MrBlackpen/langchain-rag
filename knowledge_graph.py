# LangCHain-Rag/knowledge_graph.py
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama
from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DB,
    OLLAMA_MODEL
)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

llm = ChatOllama(model=OLLAMA_MODEL)


def add_fact(entity, relation, value, workspace_id):
    with driver.session(database=NEO4J_DB) as session:
        session.run(
            """
            MERGE (e:Entity {name:$entity, workspace:$workspace})
            MERGE (v:Value {name:$value, workspace:$workspace})
            MERGE (e)-[:RELATION {type:$relation}]->(v)
            """,
            entity=entity,
            relation=relation,
            value=value,
            workspace=workspace_id
        )


def extract_and_store_facts(text, workspace_id):
    prompt = f"""
    Extract factual triples.

    Format:
    Entity | Relation | Value

    Text:
    {text}

    Return only structured lines.
    """

    response = llm.invoke(prompt).content
    lines = response.split("\n")

    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                add_fact(parts[0], parts[1], parts[2], workspace_id)


def extract_entity(query):
    prompt = f"""
    Extract ONLY the main entity name.
    Remove numbers, dates, extra words.

    Question:
    {query}

    Return only entity name.
    """

    return llm.invoke(prompt).content.strip()


def query_kg(entity, workspace_id):
    try:
        with driver.session(database=NEO4J_DB) as session:
            result = session.run(
                """
                MATCH (e:Entity {workspace:$workspace})
                WHERE toLower(e.name) CONTAINS toLower($entity)
                MATCH (e)-[r]->(v)
                RETURN e.name AS entity,
                       r.type AS relation,
                       v.name AS value
                """,
                entity=entity,
                workspace=workspace_id
            )

            return [record.data() for record in result]

    except Exception:
        return []