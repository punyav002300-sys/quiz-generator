# AI Study Assistant — CrewAI & Pinecone

An AI-powered study assistant that lets a user upload study material (PDF)
and ask questions about it. Three specialized CrewAI agents retrieve,
analyze, and verify the answer before it's shown to the user.

## Architecture

```
User uploads PDF
    ↓
Extract document text        (utils.py — pypdf)
    ↓
Generate embeddings          (vector_store.py — sentence-transformers)
    ↓
Pinecone                     (vector_store.py — stores + semantic search)

                              CrewAI Crew (agents.py)
                              Research Agent
                                  ↓
                              Analysis Agent
                                  ↓
                              Review Agent
                                  ↓
                          Final Answer → User
```

## Agent roles

| Agent | Responsibility |
|---|---|
| **Research Agent** | Searches the knowledge base for relevant information |
| **Analysis Agent** | Analyzes the retrieved information and drafts an answer |
| **Review Agent** | Checks whether the answer is supported by the retrieved content |

## Tech stack

- **Ollama LLM** — local LLM for the CrewAI agents
- **CrewAI** — manages agent roles, tasks, and communication
- **LangChain** — LLM wrapper (`langchain_community.llms.Ollama`)
- **Pinecone** — stores document embeddings and performs semantic search
- **Sentence-Transformers** — generates embeddings locally
- **Flask** — web server and API
- **HTML & CSS** — front end
- **Python** — everything glues together

## Project structure

```
ai_study_assistant/
├── app.py              # Flask routes: /, /upload, /ask, /health
├── agents.py           # CrewAI crew: Research, Analysis, Review agents
├── vector_store.py      # Pinecone storage + semantic search
├── utils.py             # PDF text extraction + chunking
├── templates/
│   └── index.html       # Upload + ask UI
├── static/
│   └── style.css
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Install and run Ollama** (for the LLM)
   ```bash
   # https://ollama.com
   ollama pull llama3
   ollama serve
   ```

3. **Get a Pinecone API key**
   - Sign up at https://www.pinecone.io
   - Create an API key, then copy `.env.example` to `.env` and fill it in:
     ```bash
     cp .env.example .env
     ```

4. **Run the app**
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000`.

## How it works

1. **Upload** — a PDF is uploaded, text is extracted page by page, split
   into overlapping word chunks, embedded with `all-MiniLM-L6-v2`, and
   upserted into a Pinecone index.
2. **Ask** — a question is embedded and used to query Pinecone for the
   top-k most similar chunks.
3. **Crew pipeline**:
   - *Research Agent* summarizes what the retrieved chunks say that's
     relevant to the question.
   - *Analysis Agent* drafts an answer using only that summary.
   - *Review Agent* checks the draft against the retrieved content,
     trims unsupported claims, and returns the final answer.
4. The final answer and its source chunks are returned to the UI.

## Notes / possible extensions

- Swap `Ollama` for any other LangChain-supported LLM by editing `agents.py`.
- Add support for `.docx` / `.txt` uploads by extending `utils.py`.
- Add a `doc_id` filter to `/ask` to scope questions to one uploaded file.
- Add authentication and per-user Pinecone namespaces for multi-user use.
