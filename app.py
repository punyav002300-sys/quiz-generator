"""
RAG Chatbot
===========

A Flask web app where users upload PDFs and ask questions.

The application uses:
- Flask       -> Web server
- LangChain   -> RAG pipeline
- FAISS       -> Local vector database
- Ollama      -> Local LLM and embedding model

Ollama models:
- LLM: llama3.2:latest
- Embeddings: nomic-embed-text:latest
"""

import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

VECTORSTORE_FOLDER = os.path.join(
    BASE_DIR,
    "vectorstore"
)

# ------------------------------------------------------------
# Ollama configuration
# ------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

# LLM used for answering questions
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

# Embedding model used by FAISS
OLLAMA_EMBED_MODEL = os.environ.get(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text:latest"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    VECTORSTORE_FOLDER,
    exist_ok=True
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "dev-secret-key-change-me"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# IN-MEMORY VECTOR STORES
# ============================================================

VECTOR_STORES = {}


# ============================================================
# RAG PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are a helpful document assistant.

Answer the user's question using the retrieved CONTEXT from
the uploaded PDF documents.

IMPORTANT RULES:

1. Use the uploaded document as the primary source.
2. Do not invent information.
3. If the answer is clearly available in the CONTEXT,
   answer directly from the CONTEXT.
4. If the information is not available in the CONTEXT,
   say that the information was not found in the uploaded PDF.
5. Keep the answer clear and concise.
6. Do not claim that information is present in the document
   if it is not present.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=[
        "context",
        "question"
    ]
)


# ============================================================
# HELPER: SESSION ID
# ============================================================

def get_session_id():
    """
    Get or create a unique session ID.
    """

    if "sid" not in session:
        session["sid"] = str(
            uuid.uuid4()
        )

    return session["sid"]


# ============================================================
# HELPER: OLLAMA EMBEDDINGS
# ============================================================

def get_embeddings():
    """
    Create the Ollama embedding model.

    Uses:
        nomic-embed-text:latest
    """

    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )


# ============================================================
# HELPER: BUILD VECTOR STORE
# ============================================================

def build_vectorstore_from_pdfs(pdf_paths):
    """
    Load PDFs, split them into chunks, create embeddings,
    and build a FAISS vector store.
    """

    all_docs = []

    # --------------------------------------------------------
    # Load all PDFs
    # --------------------------------------------------------

    for path in pdf_paths:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        try:

            loader = PyPDFLoader(path)

            documents = loader.load()

            if documents:
                all_docs.extend(documents)

        except Exception as e:

            raise RuntimeError(
                f"Failed to read PDF "
                f"{os.path.basename(path)}: {e}"
            )

    if not all_docs:

        raise ValueError(
            "No readable text was found in the uploaded PDF."
        )

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(
        all_docs
    )

    # Remove empty chunks
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content
        and chunk.page_content.strip()
    ]

    if not chunks:

        raise ValueError(
            "The PDF could not be split into readable text chunks."
        )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    try:

        embeddings = get_embeddings()

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    except Exception as e:

        message = str(e)

        if (
            "404" in message
            or "not found" in message.lower()
        ):

            raise RuntimeError(
                f"Ollama embedding model "
                f"'{OLLAMA_EMBED_MODEL}' was not found.\n\n"
                f"Please run:\n"
                f"ollama pull {OLLAMA_EMBED_MODEL}"
            )

        raise RuntimeError(
            f"Failed to create document embeddings: {message}"
        )

    return vectorstore


# ============================================================
# HELPER: CREATE QA CHAIN
# ============================================================

def get_qa_chain(vectorstore):
    """
    Create the RetrievalQA chain using llama3.2:latest.
    """

    try:

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not initialize Ollama LLM: {e}"
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": QA_PROMPT
        },
        return_source_documents=True,
    )

    return chain


# ============================================================
# ROUTE: HOME
# ============================================================

@app.route("/")
def index():
    """
    Display the main page.
    """

    return render_template(
        "index.html"
    )


# ============================================================
# ROUTE: UPLOAD PDF
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    sid = get_session_id()

    # --------------------------------------------------------
    # Get uploaded files
    # --------------------------------------------------------

    files = request.files.getlist(
        "pdfs"
    )

    if not files:

        return jsonify({
            "success": False,
            "message": "No PDF files selected."
        }), 400

    valid_files = [
        f
        for f in files
        if f
        and f.filename
        and f.filename.lower().endswith(".pdf")
    ]

    if not valid_files:

        return jsonify({
            "success": False,
            "message": "Please upload at least one valid PDF file."
        }), 400

    # --------------------------------------------------------
    # Session upload folder
    # --------------------------------------------------------

    session_folder = os.path.join(
        UPLOAD_FOLDER,
        sid
    )

    os.makedirs(
        session_folder,
        exist_ok=True
    )

    saved_paths = []

    # --------------------------------------------------------
    # Save PDFs
    # --------------------------------------------------------

    for file in valid_files:

        # Use a safe filename
        filename = os.path.basename(
            file.filename
        )

        filepath = os.path.join(
            session_folder,
            filename
        )

        try:

            file.save(filepath)

            saved_paths.append(
                filepath
            )

        except Exception as e:

            return jsonify({
                "success": False,
                "message": (
                    f"Could not save "
                    f"{filename}: {e}"
                )
            }), 500

    if not saved_paths:

        return jsonify({
            "success": False,
            "message": "No files were saved."
        }), 400

    # --------------------------------------------------------
    # Build FAISS vector store
    # --------------------------------------------------------

    try:

        vectorstore = build_vectorstore_from_pdfs(
            saved_paths
        )

        # Store in memory
        VECTOR_STORES[sid] = vectorstore

        # ----------------------------------------------------
        # Persist vector store
        # ----------------------------------------------------

        session_vectorstore_path = os.path.join(
            VECTORSTORE_FOLDER,
            sid
        )

        # Remove previous store reference from memory only.
        # save_local will create/update the files.
        vectorstore.save_local(
            session_vectorstore_path
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "message": (
                f"Error processing PDFs: {e}"
            )
        }), 500

    return jsonify({
        "success": True,
        "message": (
            f"Processed {len(saved_paths)} PDF(s). "
            f"You can start asking questions now."
        ),
        "files": [
            os.path.basename(path)
            for path in saved_paths
        ],
    })


# ============================================================
# ROUTE: ASK QUESTION
# ============================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    sid = get_session_id()

    # --------------------------------------------------------
    # Get request data
    # --------------------------------------------------------

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "question",
        ""
    )

    if not isinstance(
        question,
        str
    ):

        return jsonify({
            "success": False,
            "message": "Invalid question."
        }), 400

    question = question.strip()

    if not question:

        return jsonify({
            "success": False,
            "message": "Please type a question."
        }), 400

    # --------------------------------------------------------
    # Get vector store from memory
    # --------------------------------------------------------

    vectorstore = VECTOR_STORES.get(
        sid
    )

    # --------------------------------------------------------
    # Load from disk if necessary
    # --------------------------------------------------------

    if vectorstore is None:

        path = os.path.join(
            VECTORSTORE_FOLDER,
            sid
        )

        if os.path.exists(
            path
        ):

            try:

                embeddings = get_embeddings()

                vectorstore = FAISS.load_local(
                    path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                VECTOR_STORES[sid] = (
                    vectorstore
                )

            except Exception as e:

                return jsonify({
                    "success": False,
                    "message": (
                        f"Could not load document "
                        f"index: {e}"
                    )
                }), 500

    # --------------------------------------------------------
    # Check vector store
    # --------------------------------------------------------

    if vectorstore is None:

        return jsonify({
            "success": False,
            "message": (
                "Please upload at least one PDF "
                "before asking questions."
            )
        }), 400

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        chain = get_qa_chain(
            vectorstore
        )

        result = chain.invoke({
            "query": question
        })

        answer = result.get(
            "result",
            ""
        )

        if not answer:

            answer = (
                "I could not generate an answer "
                "from the uploaded document."
            )

        # ----------------------------------------------------
        # Source documents
        # ----------------------------------------------------

        sources = []

        for doc in result.get(
            "source_documents",
            []
        ):

            page = doc.metadata.get(
                "page",
                "?"
            )

            source = doc.metadata.get(
                "source",
                "document"
            )

            snippet = (
                doc.page_content[:200]
                .strip()
            )

            if snippet:
                snippet += "..."

            sources.append({
                "page": page,
                "source": os.path.basename(
                    source
                ),
                "snippet": snippet
            })

    except Exception as e:

        error_message = str(e)

        # ----------------------------------------------------
        # Specific Ollama model error
        # ----------------------------------------------------

        if (
            "404" in error_message
            or "model" in error_message.lower()
            and "not found" in error_message.lower()
        ):

            return jsonify({
                "success": False,
                "message": (
                    f"Ollama model "
                    f"'{OLLAMA_MODEL}' was not found.\n\n"
                    f"Run:\n"
                    f"ollama pull {OLLAMA_MODEL}"
                )
            }), 500

        return jsonify({
            "success": False,
            "message": (
                f"Error generating answer: "
                f"{error_message}"
            )
        }), 500

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return jsonify({
        "success": True,
        "answer": answer,
        "sources": sources
    })


# ============================================================
# ROUTE: RESET
# ============================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    sid = get_session_id()

    VECTOR_STORES.pop(
        sid,
        None
    )

    return jsonify({
        "success": True,
        "message": (
            "Session reset. "
            "Upload new PDFs to start again."
        )
    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG CHATBOT")
    print("=" * 60)
    print(
        f"Ollama URL       : {OLLAMA_BASE_URL}"
    )
    print(
        f"LLM Model        : {OLLAMA_MODEL}"
    )
    print(
        f"Embedding Model  : {OLLAMA_EMBED_MODEL}"
    )
    print("=" * 60)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
