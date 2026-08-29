"""
AI-Powered Quiz Generator
==========================
Flask web application for document upload and AI quiz generation.

Run with:
    python app.py

Requires:
    - Ollama installed and running
    - llama3.2:latest model
"""

import os
import uuid
import traceback

from flask import Flask, render_template, request, jsonify

from quiz_engine import (
    process_document,
    generate_quiz_from_store,
    SUPPORTED_EXTENSIONS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_FOLDER = os.path.join(BASE_DIR, "vectorstore")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key-change-me"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

SESSION_STORE = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "document" not in request.files:
        return jsonify({
            "error": "No file part in the request."
        }), 400

    file = request.files["document"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": (
                f"Unsupported file type '{ext}'. "
                f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        }), 400

    session_id = str(uuid.uuid4())

    saved_name = f"{session_id}{ext}"

    saved_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        saved_name
    )

    file.save(saved_path)

    try:

        store_path = os.path.join(
            VECTORSTORE_FOLDER,
            session_id
        )

        num_chunks = process_document(
            saved_path,
            store_path
        )

        SESSION_STORE[session_id] = store_path

    except Exception as exc:

        traceback.print_exc()

        return jsonify({
            "error": f"Failed to process document: {exc}"
        }), 500

    return jsonify({
        "session_id": session_id,
        "filename": file.filename,
        "chunks_indexed": num_chunks,
        "message": "Document processed and indexed successfully."
    })


@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():

    data = request.get_json(
        force=True,
        silent=True
    ) or {}

    session_id = data.get("session_id")

    try:
        num_questions = int(
            data.get("num_questions", 5)
        )
    except (TypeError, ValueError):
        num_questions = 5

    difficulty = data.get(
        "difficulty",
        "medium"
    )

    topic_hint = data.get(
        "topic_hint",
        ""
    )

    if not session_id or session_id not in SESSION_STORE:

        return jsonify({
            "error": (
                "Invalid or expired session_id. "
                "Please upload a document first."
            )
        }), 400

    num_questions = max(
        1,
        min(num_questions, 20)
    )

    try:

        store_path = SESSION_STORE[session_id]

        quiz = generate_quiz_from_store(
            store_path=store_path,
            num_questions=num_questions,
            difficulty=difficulty,
            topic_hint=topic_hint,
        )

    except Exception as exc:

        traceback.print_exc()

        return jsonify({
            "error": f"Failed to generate quiz: {exc}"
        }), 500

    return jsonify({
        "quiz": quiz
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )