# Thesis Chatbot (Streamlit + Groq + RAG)

This repository contains the source code for the "Thesis Chatbot," a portfolio project demonstrating a Retrieval-Augmented Generation (RAG) implementation on an academic research paper.

## LIVE DEMO

**Try the live application here:**
[**https://huggingface.co/spaces/wikan1602/thesis-chatbot**](https://huggingface.co/spaces/wikan1602/thesis-chatbot)

---

## About This Project

This application allows users to "ask questions" directly to my undergraduate thesis. Instead of reading the full document, users can ask questions in natural language (e.g., "What is the main conclusion?") and receive relevant answers sourced directly from the text.

The bot is built on the text from the thesis: *"Asosiasi Diabetes Melitus Tipe 2 dengan Brachial-Ankle Pulse Wave Velocity dan Komplians Pembuluh Darah"* (The Association of Type 2 Diabetes Mellitus with Brachial-Ankle Pulse Wave Velocity and Vascular Compliance).

## Tech Stack

* **Application Framework:** Streamlit
* **LLM Orchestration:** LangChain
* **LLM:** Groq (using `groq/compound`)
* **Vector Database:** ChromaDB
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Hosting:** Hugging Face Spaces (via Docker)

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/wikan1602/thesis-chatbot.git](https://github.com/wikan1602/thesis-chatbot.git)
    cd thesis-chatbot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Build the vector database:**
    The original `ta_teks.txt` is included. Run the `make_db.py` script to create the local `db_ta/` vector store.
    ```bash
    python make_db.py
    ```

5.  **Set up environment variables:**
    Create a file named `.env` and add your Groq API key:
    ```
    GROQ_API_KEY="your-groq-api-key-here"
    ```

6.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
