import streamlit as st
from agent_runner import create_agent
import pandas as pd
import os
import pickle

PARQUET_DIR = "./data/parquet"
MEMORY_FILE = "./data/conversation_memory.pkl"

st.set_page_config(page_title="Olist GenAI Agent", layout="wide")
st.title("🧠 Olist E-commerce Conversational Agent with Memory")

st.sidebar.header("Demo Prompts")
st.sidebar.write("- Which product category had highest GMV in 2018?")
st.sidebar.write("- What was the average delivery time per month?")
st.sidebar.write("- Summarize recent negative reviews for electronics.")

# --- Load persistent memory if exists ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return None

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

# --- Initialize Agent once ---
if "agent" not in st.session_state:
    with st.spinner("Initializing AI agent..."):
        st.session_state.agent = create_agent()

# Load existing memory state into the agent
if os.path.exists(MEMORY_FILE):
    loaded_memory = load_memory()
    if loaded_memory:
        st.session_state.agent.memory.chat_memory = loaded_memory.chat_memory

agent = st.session_state.agent

# --- Memory Reset Option ---
if st.sidebar.button("🧹 Clear Conversation Memory"):
    try:
        st.session_state.agent.memory.clear()
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.success("✅ Conversation memory cleared!")
    except Exception as e:
        st.warning(f"Memory reset failed: {e}")

# --- Chat Interface ---
st.markdown("### 💬 Ask a Question")
query = st.text_area("Ask a question about the Olist dataset:", height=100)

if st.button("Run Query"):
    if query.strip():
        with st.spinner("Running agent..."):
            try:
                response = agent.invoke({"input": query})
                st.markdown("### 🤖 Response")
                st.write(response.get("output", response))

                # Persist memory for next turn
                save_memory(agent.memory)

            except Exception as e:
                st.error(f"⚠️ Error running agent: {e}")
    else:
        st.warning("Please enter a question first.")

st.markdown("---")
st.subheader("📊 Weekly KPI Preview")
try:
    df = pd.read_parquet(os.path.join(PARQUET_DIR, "kpi_weekly.parquet"))
    st.line_chart(df.set_index("order_purchase_timestamp")["gmv"])
except Exception as e:
    st.error("⚠️ Please run etl.py first to generate parquet files.")

st.markdown("---")
st.caption("Powered by LangChain + DuckDB + FAISS + Streamlit 🚀")

# streamlit run c:\Users\JAGADEESH\OneDrive\Desktop\Assesment\src\streamlit_app.py
