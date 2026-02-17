🧠 OlistIQ — GenAI Commerce Insight Agent  
🌍 Conversational AI for E-Commerce Analytics

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://jagadeeshwar45-tradelensai-srcstreamlit-app-mhm8sd.streamlit.app/)

OlistIQ is a GenAI-driven conversational analytics agent built to explore and analyze the Brazilian Olist E-commerce dataset.  
It combines LangChain, DuckDB, FAISS, and Google Generative AI to let users ask natural language questions and get data-driven, contextual, and multilingual insights — from SQL analytics to sentiment retrieval and real-time web lookups.

🚀 Features  
🧩 Core Intelligence

- 💬 Conversational memory: Maintains multi-turn context using ConversationBufferMemory.
- 🧠 ReAct agent reasoning: Chooses optimal tools (SQL, Wikipedia, Web Search, Vector Retrieval, etc.) step-by-step.
- 📊 Data insights via SQL: Queries denormalized DuckDB Parquet tables for GMV, delivery, RFM, and customer KPIs.
- 🔎 Semantic retrieval: Uses multilingual Sentence Transformers + FAISS to search Portuguese reviews in English.
- 🌐 Live Web Search: Integrates Tavily API for real-time info (e.g., “latest iPhone model”).
- 📘 Wikipedia Lookup: Provides background info or definitions beyond dataset scope.
- 🌍 Universal Translation: Automatically detects and translates between any languages using Google Translate.
- 🗣 Definition Lookup: Short concept summaries fetched dynamically from Wikipedia.
- 🧾 Streamlit UI: Simple interactive dashboard with live charts and conversation persistence.

🏗️ Architecture  
```
project-root/
│
├── data/
│   ├── ecommerce/                 # Raw Olist CSV files
│   ├── parquet/                   # Cleaned & joined Parquet tables (generated)
│   └── faiss_index/               # FAISS index + metadata (generated)
│
├── src/
│   ├── etl.py                     # Loads and cleans raw Olist CSVs → Parquet
│   ├── build_vectorstore.py       # Builds FAISS multilingual vector index
│   ├── agent_tools.py             # DuckDB runner, FAISS retriever, chart helper
│   ├── agent_runner.py            # Main agent logic + tool definitions
│   └── streamlit_app.py           # Streamlit interface
│
├── .env.example                   # Template for environment variables
├── requirements.txt               # Python dependencies
├── README.md                      # (You’re here)
└── .gitignore                     # Ignore envs, caches, large files
```

⚙️ Setup & Installation  
**1️⃣ Clone the Repository**
```bash
git clone https://github.com/Jagadeeshwar45/TradeLensAI.git
cd TradeLensAI
```

**2️⃣ Create a Virtual Environment**
```bash
python -m venv ass.venv
ass.venv\Scripts\activate      # On Windows
# or
source ass.venv/bin/activate   # On macOS/Linux
```

**3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**4️⃣ Configure Environment Variables**

Create a `.env` file in the project root (use `.env.example` as a reference):

```
GG_MODEL=gemini-2.5-flash
TAVILY_API_KEY=your_tavily_api_key_here
```

🧹 **Data Preparation**

Place Olist CSVs under `./data/ecommerce/`  
(Download from Kaggle – Brazilian E-commerce Olist Dataset)

Run the ETL Pipeline:
```bash
python src/etl.py
```
→ Generates clean Parquet tables under `./data/parquet/`.

Build Vector Index:
```bash
python src/build_vectorstore.py
```
→ Creates FAISS index and metadata in `./data/faiss_index/`.

💬 **Run the GenAI Agent (Streamlit App)**

Launch your interactive dashboard:
```bash
streamlit run src/streamlit_app.py
```
Then open your browser (default: http://localhost:8501).  



🌟 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jagadeeshwar45-tradelensai-srcstreamlit-app-mhm8sd.streamlit.app/)  



🧩 **Example Queries**

| Query | Description |
|-------|-------------|
| “Which product category had the highest GMV in 2018?” | Runs an SQL aggregation query. |
| “Summarize recent negative reviews for electronics.” | Uses FAISS vector retrieval. |
| “What is the meaning of GMV?” | Triggers Wikipedia definition lookup. |
| “Translate ‘O cliente está muito satisfeito com o produto.’ to English.” | Uses universal translator tool. |
| “Tell me the latest iPhone model.” | Uses Tavily live web search. |

📊 **Analytics Example (KPI Preview)**

A weekly GMV trend chart is auto-rendered in Streamlit from the `kpi_weekly.parquet` dataset.

🔧 **Tech Stack**

| Layer               | Technology                              |
|---------------------|-----------------------------------------|
| LLM Backend         | Google Gemini (langchain-google-genai)  |
| Agent Framework     | LangChain ReAct Agent                   |
| Vector DB           | FAISS                                   |
| Local Query Engine  | DuckDB                                  |
| Embeddings          | sentence-transformers multilingual model|
| Frontend            | Streamlit                               |
| Data Processing     | Pandas / NumPy                          |
| Web Search          | Tavily API / Wikipedia                  |
| Translation         | Google Translate API                    |
