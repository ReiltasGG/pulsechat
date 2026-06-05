import os
import streamlit as st
import ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseChat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #f4f6f8;
    --surface:  #ffffff;
    --border:   #e0e4ea;
    --teal:     #1a7a6e;
    --teal-lt:  #2aab9a;
    --teal-dim: #d0eeeb;
    --text:     #1a2332;
    --muted:    #6b7a90;
    --shadow:   0 1px 4px rgba(0,0,0,0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { background-color: #ffffff !important; }

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.header-strip {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 20px 0;
    border-bottom: 2px solid var(--teal);
    margin-bottom: 24px;
}
.header-logo {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: var(--teal);
}
.header-sub {
    font-size: 0.82rem;
    color: var(--muted);
}

.bubble-user {
    background: var(--teal);
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px 20%;
    font-size: 0.92rem;
    line-height: 1.5;
}
.bubble-bot {
    background: var(--surface);
    color: var(--text);
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    border: 1px solid var(--border);
    margin: 8px 20% 8px 0;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: var(--shadow);
}
.bubble-source {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 6px;
    padding-left: 4px;
}

.sidebar-title {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
}
.doc-pill {
    background: var(--teal-dim);
    color: var(--teal);
    border: 1px solid var(--teal);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    display: inline-block;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Load and index documents ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def build_vectorstore():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(BASE_DIR, "docs")

    loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

# ── Query Ollama locally ──────────────────────────────────────────────────────
def query_llm(question, context):
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question. If the answer is not in the context, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}"""

    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">💬 PulseChat</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="sidebar-title">Knowledge Base</p>', unsafe_allow_html=True)
    st.markdown("""
        <span class="doc-pill">company_handbook</span>
        <span class="doc-pill">product_faq</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">Model</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#1a7a6e;font-weight:500;">Gemma 3 4B</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#6b7a90;">via Ollama (local)</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-title">Example Questions</p>', unsafe_allow_html=True)
    examples = [
        "What is the PTO policy?",
        "How much is the home office stipend?",
        "What tools do new employees get on day 1?",
        "What are FlowDesk's pricing tiers?",
        "Does FlowDesk integrate with Salesforce?",
        "What is the 401k match?",
    ]
    for q in examples:
        st.markdown(f'<p style="font-size:0.78rem;color:#6b7a90;margin:4px 0;">• {q}</p>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-strip">
    <span class="header-logo">💬 PulseChat</span>
    <span class="header-sub">AI assistant powered by Nova Tech's knowledge base</span>
</div>
""", unsafe_allow_html=True)

# ── Init chat history ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Load vectorstore ──────────────────────────────────────────────────────────
vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        sources = msg.get("sources", [])
        source_text = ""
        if sources:
            names = list(set([
                os.path.basename(s).replace(".txt", "").replace("_", " ").title()
                for s in sources
            ]))
            source_text = f'<div class="bubble-source">Sources: {", ".join(names)}</div>'
        st.markdown(
            f'<div class="bubble-bot">{msg["content"]}{source_text}</div>',
            unsafe_allow_html=True
        )

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about Nova Tech or FlowDesk...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="bubble-user">{question}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        source_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in source_docs])
        answer = query_llm(question, context)
        sources = [doc.metadata.get("source", "") for doc in source_docs]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    source_names = list(set([
        os.path.basename(s).replace(".txt", "").replace("_", " ").title()
        for s in sources
    ]))
    source_text = f'<div class="bubble-source">Sources: {", ".join(source_names)}</div>' if source_names else ""
    st.markdown(
        f'<div class="bubble-bot">{answer}{source_text}</div>',
        unsafe_allow_html=True
    )