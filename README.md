# 02 - RAG Chatbot

A web application where users upload PDFs and ask questions. The AI answers
using **retrieved context from your documents** (RAG) combined with the
**general knowledge of a local LLM**.

**Stack:** Ollama (LLM) · LangChain (orchestration) · FAISS (vector search) ·
Flask + HTML/CSS/JS (web app)

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

## 2. Pull the models Ollama will use

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Make sure Ollama is running in the background (`ollama serve`, or it may
already run automatically after install).

## 3. Set up the Python environment

```bash
cd rag-chatbot
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run the app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

## 5. How to use it

1. Click "Choose files", select one or more PDFs, click **Upload & Process**.
   The app loads the PDFs, splits them into chunks, embeds them with
   `nomic-embed-text`, and stores the vectors in a local FAISS index.
2. Type a question in the chat box and hit **Ask**.
3. The app retrieves the most relevant chunks from your PDFs and asks the
   `llama3` model (via Ollama) to answer using that context, falling back to
   its own general knowledge when the documents don't fully cover it.
4. Answers show the source pages used, so you can verify them.
5. Click **Reset session** to clear the current PDFs and start over.

## Project structure

```
rag-chatbot/
├── app.py                # Flask routes + LangChain/FAISS/Ollama logic
├── requirements.txt
├── templates/
│   └── index.html        # Chat UI
├── static/
│   ├── style.css
│   └── script.js
├── uploads/               # Uploaded PDFs get stored here (per session)
└── vectorstore/           # Saved FAISS indexes (per session), persisted to disk
```

## Notes / things you may want to change for your submission

- The chat model defaults to `llama3` and embeddings to `nomic-embed-text`.
  You can swap models by setting the environment variables `OLLAMA_MODEL`
  and `OLLAMA_EMBED_MODEL` before running, or by editing the defaults near
  the top of `app.py`.
- Each browser session gets its own FAISS index (via a Flask session
  cookie), so multiple users won't mix up documents.
- If you get a connection error, double check `ollama serve` is running and
  that you've pulled both models (step 2).
