"""
quiz_engine.py

AI-Powered Quiz Generator Core Engine

Features:
1. Load PDF, DOCX and TXT documents
2. Split documents into chunks
3. Generate embeddings using Ollama nomic-embed-text
4. Store embeddings in FAISS
5. Retrieve relevant document content
6. Generate MCQs using Ollama llama3.2:latest
"""

import os
import re
import json
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

# LLM used for generating quiz questions
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

# Separate embedding model
OLLAMA_EMBED_MODEL = os.environ.get(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text:latest"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx"
}

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


# ============================================================
# STEP 1: LOAD DOCUMENT
# ============================================================

def load_document(file_path: str):
    """
    Load PDF, TXT or DOCX document.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in _LOADERS:
        raise ValueError(
            f"Unsupported file extension: {extension}. "
            f"Supported formats: PDF, TXT, DOCX"
        )

    try:
        loader_class = _LOADERS[extension]
        loader = loader_class(file_path)

        documents = loader.load()

    except Exception as e:
        raise RuntimeError(
            f"Failed to load document: {str(e)}"
        )

    if not documents:
        raise ValueError(
            "No readable content found in the document."
        )

    return documents


# ============================================================
# STEP 2: CHUNK DOCUMENT
# ============================================================

def chunk_documents(documents) -> List[Any]:
    """
    Split documents into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(documents)

    # Remove empty chunks
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content
        and chunk.page_content.strip()
    ]

    return chunks


# ============================================================
# STEP 3: OLLAMA EMBEDDINGS
# ============================================================

def get_embeddings():
    """
    Create Ollama embedding model.

    IMPORTANT:
    Embeddings use nomic-embed-text.
    Quiz generation uses llama3.2:latest.
    """

    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )


# ============================================================
# STEP 4: PROCESS DOCUMENT
# ============================================================

def process_document(
    file_path: str,
    store_path: str
) -> int:
    """
    Load document, chunk it, generate embeddings,
    create FAISS vector store and save it.

    Returns:
        Number of chunks indexed.
    """

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    documents = load_document(file_path)

    if not documents:
        raise ValueError(
            "No readable text found in uploaded document."
        )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "Document could not be split into chunks."
        )

    # --------------------------------------------------------
    # Embeddings
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
                f"Run:\n"
                f"ollama pull {OLLAMA_EMBED_MODEL}"
            )

        raise RuntimeError(
            f"Failed to create embeddings: {message}"
        )

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(store_path) or ".",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS
    # --------------------------------------------------------

    vectorstore.save_local(store_path)

    return len(chunks)


# ============================================================
# STEP 5: LOAD VECTOR STORE
# ============================================================

def load_vectorstore(
    store_path: str
) -> FAISS:
    """
    Load previously saved FAISS vector store.
    """

    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Vector store not found: {store_path}"
        )

    try:

        embeddings = get_embeddings()

        vectorstore = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:

        message = str(e)

        if (
            "404" in message
            or "not found" in message.lower()
        ):
            raise RuntimeError(
                f"Ollama embedding model "
                f"'{OLLAMA_EMBED_MODEL}' was not found.\n\n"
                f"Run:\n"
                f"ollama pull {OLLAMA_EMBED_MODEL}"
            )

        raise RuntimeError(
            f"Failed to load FAISS vector store: {message}"
        )


# ============================================================
# STEP 6: MCQ PROMPT
# ============================================================

_MCQ_PROMPT = PromptTemplate(
    input_variables=[
        "context",
        "num_questions",
        "difficulty",
        "topic_hint"
    ],

    template="""
You are an expert exam question generator.

Create multiple-choice questions using ONLY the information
provided in the CONTEXT.

Generate exactly {num_questions} questions.

Difficulty:
{difficulty}

{topic_hint}

RULES:

1. Use ONLY information from the CONTEXT.
2. Do not use outside knowledge.
3. Each question must have exactly 4 options.
4. Options must be A, B, C and D.
5. Exactly one option must be correct.
6. Wrong options should be plausible.
7. Do not make obviously silly options.
8. Do not repeat questions.
9. Questions must be clear and educational.
10. Do not provide explanations.
11. Do not provide text outside the JSON.
12. Return ONLY valid JSON.

Use exactly this JSON structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "A"
        }}
    ]
}}

CONTEXT:

{context}
"""
)


# ============================================================
# STEP 7: GET OLLAMA LLM
# ============================================================

def _get_llm(
    temperature: float = 0.4
) -> ChatOllama:
    """
    Create Ollama LLM for quiz generation.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature
    )


# ============================================================
# STEP 8: EXTRACT JSON
# ============================================================

def _extract_json(
    raw_text: str
) -> Dict[str, Any]:
    """
    Extract JSON from model response.

    Handles:
    - normal JSON
    - JSON inside markdown fences
    - extra text surrounding JSON
    """

    if not raw_text:
        raise ValueError(
            "Ollama returned an empty response."
        )

    text = raw_text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    # Locate JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Model response did not contain valid JSON."
        )

    json_text = text[start:end + 1]

    try:

        return json.loads(json_text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Invalid JSON returned by Ollama: {str(e)}"
        )


# ============================================================
# STEP 9: GATHER CONTEXT
# ============================================================

def _gather_context(
    vectorstore: FAISS,
    topic_hint: str,
    k: int = 8
) -> str:
    """
    Retrieve document chunks.

    If topic is provided:
        similarity search is used.

    If topic is empty:
        representative chunks are selected.
    """

    if topic_hint and topic_hint.strip():

        docs = vectorstore.similarity_search(
            topic_hint.strip(),
            k=k
        )

    else:

        try:

            all_docs = list(
                vectorstore.docstore._dict.values()
            )

        except Exception:
            all_docs = []

        if not all_docs:
            return ""

        step = max(
            1,
            len(all_docs) // k
        )

        docs = all_docs[::step][:k]

    # Remove empty documents
    docs = [
        doc
        for doc in docs
        if doc.page_content
        and doc.page_content.strip()
    ]

    if not docs:
        return ""

    return "\n\n---\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# STEP 10: VALIDATE MCQ
# ============================================================

def _validate_question(
    question: Dict[str, Any]
) -> bool:
    """
    Validate one MCQ.
    """

    if not isinstance(question, dict):
        return False

    # Question
    question_text = question.get(
        "question"
    )

    if not isinstance(
        question_text,
        str
    ):
        return False

    if not question_text.strip():
        return False

    # Options
    options = question.get(
        "options"
    )

    if not isinstance(
        options,
        dict
    ):
        return False

    required_options = {
        "A",
        "B",
        "C",
        "D"
    }

    if set(options.keys()) != required_options:
        return False

    # Check every option
    for key in required_options:

        if not isinstance(
            options[key],
            str
        ):
            return False

        if not options[key].strip():
            return False

    # Correct answer
    correct_answer = question.get(
        "correct_answer"
    )

    if correct_answer not in required_options:
        return False

    return True


# ============================================================
# STEP 11: REMOVE DUPLICATES
# ============================================================

def _remove_duplicates(
    questions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate questions.
    """

    unique_questions = []
    seen = set()

    for question in questions:

        normalized = re.sub(
            r"\s+",
            " ",
            question["question"]
            .strip()
            .lower()
        )

        if normalized not in seen:

            seen.add(normalized)
            unique_questions.append(
                question
            )

    return unique_questions


# ============================================================
# STEP 12: GENERATE QUIZ
# ============================================================

def generate_quiz_from_store(
    store_path: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    topic_hint: str = ""
) -> List[Dict[str, Any]]:
    """
    Generate MCQs from a saved FAISS vector store.
    """

    # --------------------------------------------------------
    # Validate number of questions
    # --------------------------------------------------------

    try:

        num_questions = int(
            num_questions
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "Number of questions must be an integer."
        )

    if num_questions < 1:
        raise ValueError(
            "Number of questions must be at least 1."
        )

    if num_questions > 50:
        raise ValueError(
            "Maximum 50 questions can be generated."
        )

    # --------------------------------------------------------
    # Validate difficulty
    # --------------------------------------------------------

    difficulty = str(
        difficulty
    ).strip().lower()

    allowed_difficulties = {
        "easy",
        "medium",
        "hard"
    }

    if difficulty not in allowed_difficulties:

        raise ValueError(
            "Difficulty must be easy, medium, or hard."
        )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    vectorstore = load_vectorstore(
        store_path
    )

    # --------------------------------------------------------
    # Retrieve context
    # --------------------------------------------------------

    context = _gather_context(
        vectorstore,
        topic_hint,
        k=8
    )

    if not context.strip():

        raise ValueError(
            "Could not retrieve content from "
            "the indexed document."
        )

    # --------------------------------------------------------
    # Create LLM
    # --------------------------------------------------------

    llm = _get_llm(
        temperature=0.4
    )

    # --------------------------------------------------------
    # Topic instruction
    # --------------------------------------------------------

    if topic_hint and topic_hint.strip():

        topic_instruction = (
            "Focus specifically on the topic: "
            + topic_hint.strip()
        )

    else:

        topic_instruction = (
            "Cover important concepts from the document."
        )

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = _MCQ_PROMPT.format(
        context=context,
        num_questions=num_questions,
        difficulty=difficulty,
        topic_hint=topic_instruction
    )

    # --------------------------------------------------------
    # Call Ollama
    # --------------------------------------------------------

    try:

        response = llm.invoke(
            prompt
        )

    except Exception as e:

        message = str(e)

        if "404" in message:

            raise RuntimeError(
                f"Ollama model '{OLLAMA_MODEL}' "
                f"was not found.\n\n"
                f"Run:\n"
                f"ollama pull {OLLAMA_MODEL}"
            )

        raise RuntimeError(
            f"Failed to generate quiz: {message}"
        )

    # --------------------------------------------------------
    # Get response text
    # --------------------------------------------------------

    if hasattr(
        response,
        "content"
    ):

        raw_text = response.content

    else:

        raw_text = str(response)

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    parsed = _extract_json(
        raw_text
    )

    questions = parsed.get(
        "questions",
        []
    )

    if not isinstance(
        questions,
        list
    ):

        raise ValueError(
            "Invalid questions format returned by model."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    valid_questions = []

    for question in questions:

        if _validate_question(
            question
        ):

            valid_questions.append(
                question
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    valid_questions = _remove_duplicates(
        valid_questions
    )

    # --------------------------------------------------------
    # Final check
    # --------------------------------------------------------

    if not valid_questions:

        raise ValueError(
            "Ollama did not return any valid MCQs."
        )

    # --------------------------------------------------------
    # Return requested number
    # --------------------------------------------------------

    return valid_questions[
        :num_questions
    ]
