import os
import re
import difflib
import time
import json
from streamlit import text
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from agent_tools import DuckDBRunner, FaissRetriever, df_to_plot_png
from tavily import TavilyClient
from googletrans import Translator
from pathlib import Path
import requests
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"  # keep if behind proxy/ssl interception


PARQUET_DIR = "./data/parquet"
FAISS_DIR = "./data/faiss_index"

DEFAULT_MODEL = os.getenv("GG_MODEL", "gemini-2.5-flash")

def get_llm():
    return ChatGoogleGenerativeAI(
        temperature=0.0,          
        model=DEFAULT_MODEL,
    )

def extract_table_candidates_from_sql(sql: str):
    """Extract simple table names referenced after FROM / JOIN (naive, but effective)."""
    pattern = re.compile(r'\bFROM\s+([^\s,;]+)|\bJOIN\s+([^\s,;]+)', re.IGNORECASE)
    matches = pattern.findall(sql)
    candidates = []
    for m in matches:
        tbl = m[0] or m[1]
        # strip alias if present ("table AS t" or "table t")
        tbl = tbl.split('.')[ -1 ]  # allow schema.table -> table
        tbl = tbl.split()[0]
        tbl = tbl.strip('"').strip("'")
        candidates.append(tbl.lower())
    return list(dict.fromkeys(candidates))

def pretty_cols(cols):
    return ", ".join(cols)

def build_system_message(table_schemas):
    """
    table_schemas: dict table -> list(columns)
    returns a clear system instruction with schema details and hard constraints.
    """
    lines = [
        "You are a data analyst assistant for the Brazilian Olist dataset.",
        "Use the provided tools to run read-only SQL queries on the DuckDB tables.",
        "Important constraints:",
        "- Only use the listed tables. Do NOT invent table names.",
        "- If SQL execution fails because a table/column isn't found, call the 'list_tables' or 'describe_table' tool to inspect schema.",
        "- Use 'order_items_denorm' for product, category, and price information (it is denormalized).",
        "- SQL must be DuckDB-compatible.",
        "- Always consider conversation context (chat_history) for follow-up questions.",
        "- If a follow-up is ambiguous (e.g., 'What about 2018?'), assume it refines the previous user intent unless the user explicitly changes intent.",
        "",
        "Available tables and columns:"
    ]
    for t, cols in table_schemas.items():
        lines.append(f"- {t}: {pretty_cols(cols)}")
    lines.append("")
    lines.append("If you propose an SQL query that references unknown tables or unknown columns, I'll return a clear validation error and you should correct the query.")
    lines.append("- If a query is unrelated to the Olist dataset or lacks relevant information in tables or FAISS index, use 'wiki_lookup' or 'web_search' to answer using external knowledge sources.")
    return "\n".join(lines)

from langchain.agents import create_react_agent, AgentExecutor

from pathlib import Path
import subprocess
import os

import gdown

def create_agent():
    base_dir = Path(__file__).resolve().parent.parent
    index_dir = base_dir / "data" / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    meta_path  = index_dir / "docs_meta.pkl"

    # --- Google Drive file IDs ---
    FAISS_INDEX_ID = "1pOx2dcv7i7xR3BSs9r8GLj-UrXd13f3z"
    FAISS_META_ID  = "1-MqxsGV6-nC22lWcnywArM61DEJGW_l5"

    # --- Download if missing ---
    if not index_path.exists() or not meta_path.exists():
        print("⬇️ Downloading FAISS index from Google Drive...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={FAISS_INDEX_ID}", str(index_path), quiet=False)
            gdown.download(f"https://drive.google.com/uc?id={FAISS_META_ID}", str(meta_path), quiet=False)

            # Sanity check
            if index_path.stat().st_size < 50000 or meta_path.stat().st_size < 5000:
                raise RuntimeError("Downloaded files too small — likely Google Drive blocked direct download.")

            print("✅ FAISS index downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download FAISS index: {e}")
            raise RuntimeError("Could not fetch FAISS index from Drive.")


    db = DuckDBRunner(str(base_dir / "data" / "parquet"))
    retriever = FaissRetriever(str(index_dir))

    # Register available tables
    table_list = db.conn.execute("SHOW TABLES;").fetchdf()["name"].tolist()
    table_schemas = {
        t: db.conn.execute(f"PRAGMA table_info({t})").fetchdf()["name"].str.lower().tolist()
        for t in table_list
    }

    # Build schema-aware system instruction
    system_msg = build_system_message(table_schemas)

    # === TOOL DEFINITIONS ===
    def list_tables_tool(_q: str = ""):
        return "\n".join(table_list)

    def describe_table_tool(table_name: str):
        if not table_name:
            return "Usage: describe_table <table_name>"
        t = table_name.strip().lower()
        if t not in table_schemas:
            suggestion = difflib.get_close_matches(t, table_schemas.keys(), n=3)
            return f"Unknown table '{t}'. Did you mean: {', '.join(suggestion) if suggestion else 'none'}?"
        cols = table_schemas[t]
        return f"Columns for {t}: {', '.join(cols)}"

    def validate_sql(sql: str):
        referenced = extract_table_candidates_from_sql(sql.lower())
        unknown = [t for t in referenced if t not in table_schemas]
        if unknown:
            suggestions = {t: difflib.get_close_matches(t, table_schemas.keys(), n=1) for t in unknown}
            return False, {"error": "unknown_table", "unknown_tables": unknown, "suggestions": suggestions}
        return True, {}

    def sql_func(q: str):
        is_valid, info = validate_sql(q)
        if not is_valid:
            return f"SQL validation error: {json.dumps(info)}\nHint: call 'list_tables' or 'describe_table <table>' to inspect schema."
        try:
            df = db.run(q)
            try:
                return df.head(10).to_markdown()
            except Exception:
                return df.head(10).to_string(index=False)
        except Exception as e:
            return f"SQL execution error: {e}"

    def vec_func(q: str):
        docs = retriever.retrieve(q)
        if not docs:
            return "No matching documents found."
        return "\n\n".join([f"{d.get('meta','')}: {d['text'][:400]}" for d in docs])
    
    def wiki_lookup(q: str):
        """Fetch summary information from Wikipedia for broader product or concept queries."""
        import wikipedia
        try:
            q_norm = q.strip().lower()
            # Normalize obvious category names
            if "beleza" in q_norm and "saude" in q_norm:
                q = "Beauty and health industry"
            elif "eletronicos" in q_norm:
                q = "Consumer electronics industry"
            elif "esporte" in q_norm:
                q = "Sporting goods"
            # Add more mappings as needed
            results = wikipedia.search(q)
            if not results:
                return f"No Wikipedia results found for '{q}'."
            summary = wikipedia.summary(results[0], sentences=4, auto_suggest=True)
            return f"📘 Wikipedia Summary ({results[0]}):\n{summary}"
        except Exception as e:
            return f"Wikipedia lookup error: {e}"
        
    def web_search(q: str):
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(q)
        return "🌐 Web Search Results:\n" + "\n".join([r["content"] for r in results["results"][:5]])

    def decide_tool_for_query(q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ["latest", "current", "recent", "today", "price", "cost", "new", "released", "launch", "trend", "news"]):
            return "web_search"
        if any(k in ql for k in ["history", "origin", "who is", "founded", "biography"]):
            return "wiki_lookup"
        if any(k in ql for k in ["gmv", "order", "delivery", "customer", "payment", "sales", "revenue"]):
            return "sql_runner"
        if any(k in ql for k in ["review", "feedback", "sentiment", "description"]):
            return "vector_retriever"
        if any(k in ql for k in ["definition", "meaning", "define", "what is"]):
            return "definition_lookup"
        if any(k in ql for k in ["translate", "language", "in english"]):
            return "translator"
        return "web_search"
        

    def definition_lookup(term: str):
        try:
            results = wikipedia.search(term)
            if not results:
                return f"No definition found for '{term}'."
            summary = wikipedia.summary(results[0], sentences=2)
            return f"📘 Definition of {term}:\n{summary}"
        except Exception as e:
            return f"Definition lookup failed: {e}"
        
        
        
    def translate_text(text: str, target_lang: str = None):
        try:
            translator = Translator(service_urls=["translate.googleapis.com"])
            detection = translator.detect(text)
            detected_lang = getattr(detection, "lang", "unknown")

            # Auto-target: if text is English → Portuguese, else → English
            if not target_lang:
                target_lang = "en" if detected_lang != "en" else "pt"

            translation = translator.translate(text, dest=target_lang)
            return (
                f"🌍 Detected language: {detected_lang}\n"
                f"🎯 Target language: {target_lang}\n"
                f"💬 Translation: {translation.text}"
           )
        except Exception as e:
            return f"⚠️ Translation failed: {e}"
        
            
    # Register tools
    tools = [
        Tool(name="sql_runner", func=sql_func, description="Run validated read-only SQL queries on DuckDB."),
        Tool(name="vector_retriever", func=vec_func, description="Retrieve product reviews or descriptions using vector similarity."),
        Tool(name="list_tables", func=list_tables_tool, description="List available DuckDB tables."),
        Tool(name="describe_table", func=describe_table_tool, description="Describe columns of a table. Usage: describe_table <table_name>"),
        Tool(name="wiki_lookup", func=wiki_lookup, description="Retrieve factual information or product background details from Wikipedia when outside Olist data."),
        Tool(name="web_search", func=web_search, description="Perform live web searches for recent or real-time information."),
        Tool(name="definition_lookup", func=definition_lookup, description="Get concise definitions of terms."),
        Tool(name="translator", func=translate_text, description="Translate text between any languages. Input as raw text or JSON: {\"text\": \"...\", \"target_lang\": \"...\"}."),
        Tool(name="decide_tool_for_query", func=decide_tool_for_query, description="Decide which tool to use for a given query.")
   ]

    # === LLM & MEMORY ===
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # === Build ReAct Agent ===
    react_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        system_msg
        + "\n\nYou have access to the following tools:\n"
        "{tools}\n\n"
        "⚙ STRICT FORMAT RULES:\n"
        "You must follow this EXACT sequence when reasoning:\n"
        "Thought: <1 short line>\n"
        "Action: <one of [{tool_names}]>\n"
        "Action Input: <the query to send to that tool>\n"
        "Observation: <result returned by the tool>\n"
        "Repeat Thought/Action/Observation as needed.\n"
        "When ready to answer the user, respond ONLY with:\n"
        "Final Answer: <your final concise response>\n\n"
        "Example:\n"
        "Thought: I should look up the latest iPhone model.\n"
        "Action: web_search\n"
        "Action Input: latest iPhone model and price\n"
        "Observation: The search result says iPhone 16 was released.\n"
        "Final Answer: The latest iPhone model is the iPhone 16.\n"

    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

    react_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    # === Agent Executor (memory integrated) ===
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=react_agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor


if __name__ == "__main__":
    agent = create_agent()
    q = "Which product category had the highest GMV overall?"
    try:
        response = agent.invoke({"input": q})
        print("\n=== Final Answer ===")
        print(response["output"])
    except Exception as e:
        print("Agent run error:", e)
