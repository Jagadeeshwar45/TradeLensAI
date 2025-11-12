import streamlit as st
from agent_runner import create_agent
import pandas as pd
import os
import pickle

PARQUET_DIR = "./data/parquet"
MEMORY_FILE = "./data/conversation_memory.pkl"

# --- Page Config ---
st.set_page_config(
    page_title="🛍️ OlistIQ — E-Commerce Insight Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for styling ---
st.markdown("""
<style>
/* Background and text */
body {
    background-color: #0E1117;
    color: #FAFAFA;
}
[data-testid="stSidebar"] {
    background-color: #161A1D;
    color: #EAEAEA;
}
h1, h2, h3, h4 {
    color: #F8F9FA;
}
.stButton>button {
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
}

/* 🟢 Run Query button (green theme) */
div.stButton > button:first-child {
    background-color: #00C853;
    color: white;
}
div.stButton > button:first-child:hover {
    background-color: #00E676;
    color: #FFFFFF;
    transition: 0.2s;
}

/* 🔴 Sidebar Clear Memory button (red theme) */
section[data-testid="stSidebar"] .stButton>button {
    background-color: #FF4B4B !important;
    color: white !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background-color: #FF6B6B !important;
}

textarea {
    border-radius: 8px !important;
}
hr {
    border: 1px solid #2C2C2C;
}
</style>
""", unsafe_allow_html=True)


# --- Title & Header ---
st.markdown("<h1 style='text-align: center;'>🧠 OlistIQ — E-Commerce Conversational Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color:#CCCCCC;'>Ask questions, explore insights, and analyze customer trends — powered by LangChain, DuckDB & FAISS ⚡</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("💡 Demo Prompts")
st.sidebar.markdown("""
- 🛒 Which product category had **highest GMV in 2018**?  
- 🚚 What was the **average delivery time per month**?  
- 💬 Summarize **recent negative reviews** for electronics.
""")
st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear Conversation Memory"):
    try:
        if "agent" in st.session_state:
            st.session_state.agent.memory.clear()
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.sidebar.success("✅ Memory cleared successfully!")
    except Exception as e:
        st.sidebar.warning(f"Memory reset failed: {e}")

# --- Load or Initialize Agent ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return None

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

if "agent" not in st.session_state:
    with st.spinner("🚀 Initializing your AI Agent..."):
        st.session_state.agent = create_agent()

if os.path.exists(MEMORY_FILE):
    mem = load_memory()
    if mem:
        st.session_state.agent.memory.chat_memory = mem.chat_memory

agent = st.session_state.agent

# --- Main Query Section ---
st.markdown("### 💬 Ask a Question")
query = st.text_area("Type your question below:", height=100, placeholder="Enter your question about Olist data...")

if st.button("💡 Run Query"):
    if query.strip():
        with st.spinner("🧠 Thinking... generating insights..."):
            try:
                response = agent.invoke({"input": query})
                st.success("✅ Query executed successfully!")
                st.markdown("### 🤖 Agent Response")
                st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:8px;'>{response.get('output', response)}</div>", unsafe_allow_html=True)
                save_memory(agent.memory)
            except Exception as e:
                st.error(f"⚠️ Error running agent: {e}")
    else:
        st.warning("Please enter a question first.")

st.markdown("---")

# --- KPI Visualization ---
st.subheader("📊 Weekly KPI Preview")
try:
    df = pd.read_parquet(os.path.join(PARQUET_DIR, "kpi_weekly.parquet"))
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    st.line_chart(
        df.set_index("order_purchase_timestamp")["gmv"],
        use_container_width=True,
    )
    st.caption("📈 Weekly GMV trend based on Olist order data.")
except Exception:
    st.error("⚠️ Please run `etl.py` first to generate parquet files.")

st.markdown("---")
st.caption("💼 Powered by LangChain · DuckDB · FAISS · Streamlit | Theme: OlistIQ Commerce Intelligence 🛍️")
