# LangChain-Rag/vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import VECTOR_DB_DIR
import streamlit as st

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


@st.cache_resource
def get_vector_db(workspace_id):
    return Chroma(
        collection_name=f"workspace_{workspace_id}",
        embedding_function=embedding,
        persist_directory=VECTOR_DB_DIR
    )


def store_conversation(user_input, ai_response, workspace_id, chat_id):
    vector_db = get_vector_db(workspace_id)

    doc = Document(
        page_content=f"User: {user_input}\nAI: {ai_response}",
        metadata={
            "workspace": workspace_id,
            "chat_id": chat_id,
            "type": "conversation"
        }
    )

    vector_db.add_documents([doc])
    vector_db.persist()


def store_document(text, workspace_id):
    vector_db = get_vector_db(workspace_id)

    chunks = text_splitter.split_text(text)

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "workspace": workspace_id,
                "type": "document"
            }
        )
        for chunk in chunks
    ]

    vector_db.add_documents(docs)
    vector_db.persist()


def search_similar(query, workspace_id, k=5):
    vector_db = get_vector_db(workspace_id)

    results = vector_db.similarity_search(
        query,
        k=k,
        filter={"workspace": workspace_id}
    )

    return "\n".join([doc.page_content for doc in results])