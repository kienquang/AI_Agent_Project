# 🤖 Autonomous AI Customer Support Agent (Enterprise-Grade)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF9900?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=for-the-badge&logo=postgresql)
![Automation](https://img.shields.io/badge/Make.com-Automation-8A2BE2?style=for-the-badge)

An End-to-End, Production-Ready AI Customer Support System built with **FastAPI, LangGraph, and PostgreSQL**. This system autonomously handles customer inquiries, retrieves company policies via RAG, and executes complex automation workflows (Google Sheets & Gmail) while maintaining strict security, persistent memory, and human oversight.

---

## 🌟 Key Architectural Features

1. **Multi-Agent Orchestration (LangGraph):**
   - **Guardrails Node:** Acts as an AI Firewall to detect and block Prompt Injection and Jailbreak attempts before processing.
   - **Semantic Router:** Intelligently routes user queries to the appropriate specialized agent (RAG or Action) without relying on fragile tool-calling.
   - **Data Extraction:** Uses `Pydantic` and `Structured Output` (Llama-3.3-70b) to extract exact entities (Name, Email, Issue) with zero hallucination.

2. **Enterprise Cloud Database (Neon PostgreSQL):**
   - **pgvector RAG:** Company policies are embedded and stored in PostgreSQL for rapid semantic search.
   - **Persistent Checkpointing:** Uses `PostgresSaver` to maintain long-term chat history and session states across multiple concurrent users.
   - **Memory Optimizer:** Automatically summarizes and prunes old chat logs in the database when the conversation exceeds limits, saving API costs and preventing context-window overflow.

3. **Human-in-the-Loop (HITL) & Automation:**
   - **Execution Pausing:** The LangGraph state machine pauses execution before triggering sensitive external APIs.
   - **Zero-Touch Pipeline:** Once approved by the user via the Streamlit UI, the system triggers a **Make.com Webhook**, which automatically logs the ticket into **Google Sheets** and sends a confirmation **Gmail**.

---

## 📊 RAG Quality Evaluation (LLM-as-a-Judge)

To ensure reliability, the RAG pipeline is evaluated using **Ragas** against ground-truth datasets.

**Latest Evaluation Metrics:**
- **Faithfulness:** `1.0 / 1.0` *(Ensures the AI only speaks based on retrieved context).*
- **Answer Relevancy:** `0.93 / 1.0` *(Measures how directly the AI answers the user's prompt).*
- **Context Precision & Recall:** `1.0 / 1.0` *(Evaluates the quality of chunks retrieved from `pgvector`).*

---

## 🏗️ System Architecture & Folder Structure

Following Software Engineering best practices, the codebase is modularized with the Separation of Concerns principle.

```text
├── core/
│   ├── __init__.py
│   ├── state.py         # AgentState and Pydantic Models
│   ├── nodes.py         # Core logic for Guard, Supervisor, RAG, and Action Nodes
│   ├── engine.py        # LangGraph StateGraph compilation and Database connection
│   └── rag_engine.py    # PostgreSQL pgvector retrieval logic
├── evaluate_rag.py      # Automated testing script using Ragas
├── upload_knowledge.py  # Data ingestion script for embeddings
├── main.py              # FastAPI endpoints
├── app_ui.py            # Streamlit Frontend with HITL interaction
├── requirements.txt
└── README.md
🚀 Installation & Setup
1. Prerequisites
Python 3.10+
A Neon.tech PostgreSQL Database
A Groq API Key
A Make.com Webhook URL

2. Environment Variables (.env)
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require

3. Quick Start
# Install dependencies
pip install -r requirements.txt

# Step 1: Ingest knowledge into PostgreSQL Vector DB
python upload_knowledge.py

# Step 2: Start the FastAPI Backend
uvicorn main:app --reload --port 8000

# Step 3: Start the Streamlit Frontend (In a new terminal)
streamlit run app_ui.py