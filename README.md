# AI-Powered Quiz Generator

An AI-powered web application that automatically analyses uploaded documents
and generates multiple-choice questions (MCQs) — each with 4 answer options
and the correct answer — to reduce the manual effort of creating candidate
assessments.

**Tech stack (as required by the assignment):**
- **Ollama LLM** — local LLM inference (generation + embeddings)
- **LangChain** — document loading, chunking, retrieval, prompting
- **FAISS** — vector database for semantic search over document chunks
- **Flask + HTML/CSS/JS** — web application and UI

---

## 1. Problem Statement

Manually writing candidate assessment questions from training material or
reference documents is slow and inconsistent. This tool lets a recruiter or
trainer upload a document (PDF / DOCX / TXT), and automatically produces a
configurable multiple-choice quiz — with correct answers pre-marked — grounded
strictly in the content of that document.

## 2. Architecture

```
                 ┌────────────────────┐
   User uploads  │                    │
   PDF/DOCX/TXT  │   Flask Backend    │
  ───────────────▶     (app.py)       │
                 │                    │
                 └─────────┬──────────┘
                           │
                 1. Load & chunk document
                    (LangChain loaders +
                     RecursiveCharacterTextSplitter)
                           │
                           ▼
                 2. Embed chunks
                    (OllamaEmbeddings)
                           │
                           ▼
                 3. Store in FAISS vector DB
                    (persisted per session)
                           │
        User requests quiz (n questions,
        difficulty, optional topic)
                           │
                           ▼
                 4. Retrieve representative
                    context chunks from FAISS
                           │
                           ▼
                 5. Prompt Ollama LLM (ChatOllama)
                    to generate MCQs as strict JSON
                           │
                           ▼
                 6. Validate & return quiz JSON
                           │
                           ▼
                 7. Render interactive quiz in
                    browser, auto-graded client-side
```

### Data flow summary

| Stage | Component | File |
|---|---|---|
| Document loading | `PyPDFLoader` / `TextLoader` / `Docx2txtLoader` | `quiz_engine.py` |
| Chunking | `RecursiveCharacterTextSplitter` | `quiz_engine.py` |
| Embeddings | `OllamaEmbeddings` | `quiz_engine.py` |
| Vector store | `FAISS` (saved to `/vectorstore/<session_id>`) | `quiz_engine.py` |
| MCQ generation | `ChatOllama` + structured JSON prompt | `quiz_engine.py` |
| API endpoints | `/upload`, `/generate_quiz` | `app.py` |
| UI | Upload form → config form → interactive quiz | `templates/index.html`, `static/` |

## 3. Project Structure

```
quiz-generator/
├── app.py                # Flask routes: /, /upload, /generate_quiz, /health
├── quiz_engine.py         # Document processing + FAISS + MCQ generation logic
├── requirements.txt
├── README.md
├── templates/
│   └── index.html         # 3-step UI: upload -> configure -> take quiz
├── static/
│   ├── style.css
│   └── script.js           # Upload, generation, rendering, client-side scoring
├── uploads/                # Uploaded source documents (created at runtime)
└── vectorstore/            # Per-session FAISS indexes (created at runtime)
```

## 4. Setup & Run Instructions

### Step 1 — Install Ollama and pull a model
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3
ollama serve        # starts the local LLM server on http://localhost:11434
```

### Step 2 — Install Python dependencies
```bash
cd quiz-generator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

### Optional environment variables
| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3` | Which pulled Ollama model to use for embeddings + generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `SECRET_KEY` | dev key | Flask session secret (set a real one in production) |

## 5. How to Use

1. **Upload** a PDF, DOCX, or TXT document. The backend extracts the text,
   splits it into overlapping chunks, embeds them, and stores them in a
   FAISS index.
2. **Configure** the quiz: number of questions (1–20), difficulty
   (easy/medium/hard), and an optional topic focus to bias retrieval toward
   a specific section of the document.
3. **Generate** — the app retrieves representative chunks from FAISS and
   prompts the LLM to write strictly document-grounded MCQs as JSON.
4. **Take the quiz** — select an answer per question and click *Submit
   Answers* to see your score, with correct/incorrect options highlighted.

## 6. Key Design Decisions

- **Session-scoped vector stores**: each upload gets its own FAISS index on
  disk keyed by a UUID, so multiple users/documents don't collide.
- **Grounded generation**: the LLM prompt explicitly restricts question
  writing to the supplied CONTEXT only, reducing hallucinated questions.
- **Structured JSON output**: the model is asked to return strict JSON
  matching a fixed schema; a regex-based extractor tolerates stray
  markdown fences some models add, and malformed questions are filtered
  out before being returned to the client.
- **Client-side grading**: since the correct answers are already known
  (returned with the quiz), scoring is done instantly in the browser
  without an extra round trip.

## 7. Possible Extensions

- Persist sessions/quizzes in a real database (SQLite/PostgreSQL) instead
  of in-memory dict + on-disk FAISS folders.
- Add authentication so recruiters can manage multiple candidate assessments.
- Export generated quizzes to PDF/CSV.
- Support multi-document uploads merged into a single vector store.
- Add explanations for each correct answer (extend the JSON schema).
