# LangChain-Rag/app.py
import streamlit as st
import uuid
import json
import os
import time
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL
from intent_classifier import classify_intent
from memory_manager import retrieve_context
from vector_store import store_conversation, store_document, get_vector_db
from document_loader import load_document
from knowledge_graph import extract_and_store_facts
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------------- BASIC CONFIG ----------------
st.set_page_config(layout="wide")
st.title("Hybrid Multi-Workspace RAG Chatbot")

llm = ChatOllama(model=OLLAMA_MODEL)

# ---------------- WORKSPACE MANAGEMENT ----------------
WORKSPACE_META_DIR = "workspace_metadata"
os.makedirs(WORKSPACE_META_DIR, exist_ok=True)

workspace_file = os.path.join(WORKSPACE_META_DIR, "workspaces.json")

if not os.path.exists(workspace_file):
    with open(workspace_file, "w") as f:
        json.dump(["default"], f)

with open(workspace_file, "r") as f:
    workspaces = json.load(f)

st.sidebar.title("Workspaces")

# Create Workspace
new_workspace_name = st.sidebar.text_input("New Workspace Name")

if st.sidebar.button("➕ Create Workspace"):
    if new_workspace_name and new_workspace_name not in workspaces:
        workspaces.append(new_workspace_name)
        with open(workspace_file, "w") as f:
            json.dump(workspaces, f)
        st.sidebar.success("Workspace created!")
        st.rerun()

workspace_id = st.sidebar.selectbox("Select Workspace", workspaces)

# Reset when workspace changes
if "current_workspace" not in st.session_state:
    st.session_state.current_workspace = workspace_id

if st.session_state.current_workspace != workspace_id:
    st.session_state.history = []
    st.session_state.loaded_chat_id = None
    st.session_state.current_workspace = workspace_id

# ---------------- CHAT MANAGEMENT ----------------
CHAT_META_DIR = "chat_metadata"
os.makedirs(CHAT_META_DIR, exist_ok=True)

meta_file = os.path.join(CHAT_META_DIR, f"{workspace_id}.json")

if not os.path.exists(meta_file):
    with open(meta_file, "w") as f:
        json.dump({}, f)

with open(meta_file, "r") as f:
    chat_metadata = json.load(f)

st.sidebar.title("Chats")

# Create new chat
if st.sidebar.button("➕ New Chat"):
    new_chat_id = str(uuid.uuid4())
    chat_metadata[new_chat_id] = "New Chat"

    with open(meta_file, "w") as f:
        json.dump(chat_metadata, f)

    st.session_state.chat_id = new_chat_id
    st.session_state.history = []
    st.session_state.loaded_chat_id = new_chat_id
    st.rerun()

# Auto-create first chat if none exist
if not chat_metadata:
    new_chat_id = str(uuid.uuid4())
    chat_metadata[new_chat_id] = "New Chat"
    with open(meta_file, "w") as f:
        json.dump(chat_metadata, f)

# Select chat
selected_chat_id = st.sidebar.radio(
    "Select Chat",
    list(chat_metadata.keys()),
    format_func=lambda x: chat_metadata[x]
)

st.session_state.chat_id = selected_chat_id

# ---------------- LOAD CHAT HISTORY SAFELY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "loaded_chat_id" not in st.session_state:
    st.session_state.loaded_chat_id = None


def load_chat_history(workspace_id, chat_id):
    vector_db = get_vector_db(workspace_id)

    try:
        results = vector_db.get(where={"chat_id": chat_id})
    except:
        return []

    history = []

    if results and "documents" in results:
        for doc in results["documents"]:
            lines = doc.split("\n")
            if len(lines) >= 2:
                user_line = lines[0].replace("User: ", "")
                ai_line = lines[1].replace("AI: ", "")
                history.append(("User", user_line))
                history.append(("AI", ai_line))

    return history


# Load only if switching chats
if st.session_state.loaded_chat_id != selected_chat_id:
    st.session_state.history = load_chat_history(workspace_id, selected_chat_id)
    st.session_state.loaded_chat_id = selected_chat_id

# ---------------- FILE UPLOAD (Workspace Level) ----------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    text = load_document(uploaded_file)
    if text:
        store_document(text, workspace_id)
        extract_and_store_facts(text, workspace_id)
        success_placeholder = st.empty()
        success_placeholder.success("Stored in Vector DB + Knowledge Graph!")

        time.sleep(2)  # seconds
        success_placeholder.empty()

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask something...")

if user_input:

    intent = classify_intent(user_input)

    # Retrieve context across ALL chats in workspace
    context = retrieve_context(user_input, intent, workspace_id)

    messages = []

    # System instruction
    messages.append(
        SystemMessage(
            content="You are a helpful AI assistant. Use provided context when relevant."
        )
    )

    # Add intra-chat memory
    for role, msg in st.session_state.history[-6:]:
        if role == "User":
            messages.append(HumanMessage(content=msg))
        else:
            messages.append(AIMessage(content=msg))

    # Inject inter-chat / workspace context
    if context:
        messages.append(
            SystemMessage(content=f"Relevant Workspace Context:\n{context}")
        )

    # Current question
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages).content

    # Rename chat if first message
    if chat_metadata[selected_chat_id] == "New Chat":
        chat_metadata[selected_chat_id] = user_input[:40]
        with open(meta_file, "w") as f:
            json.dump(chat_metadata, f)

    # Store in session
    st.session_state.history.append(("User", user_input))
    st.session_state.history.append(("AI", response))

    # Persist conversation
    store_conversation(
        user_input,
        response,
        workspace_id,
        selected_chat_id
    )

# ---------------- DISPLAY CHAT ----------------
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)