"""
04 - AI Study Assistant Using CrewAI & Pinecone
-------------------------------------------------
Flask entrypoint. Handles PDF upload (extract -> chunk -> embed -> store in
Pinecone) and question answering (Research Agent -> Analysis Agent ->
Review Agent -> Final Answer), matching the architecture:

User uploads PDF
    -> Extract document text
    -> Generate embeddings
    -> Pinecone
                                CrewAI Crew
                                Research Agent
                                    -> Analysis Agent
                                        -> Review Agent
                                            -> Final Answer -> User
"""

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils import extract_text_from_pdf, chunk_text, generate_doc_id
from vector_store import add_documents, index_is_configured
from agents import StudyAssistantCrew

load_dotenv()

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_crew = None


def get_crew():
    global _crew
    if _crew is None:
        _crew = StudyAssistantCrew()
    return _crew


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "pinecone_configured": index_is_configured(),
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        text = extract_text_from_pdf(filepath)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to read PDF: {exc}"}), 400

    if not text.strip():
        return jsonify({"error": "Could not extract any text from this PDF"}), 400

    chunks = chunk_text(text)
    doc_id = generate_doc_id()

    try:
        count = add_documents(chunks, doc_id, source_name=filename)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to index document: {exc}"}), 500

    return jsonify({
        "message": f"Uploaded and indexed '{filename}' successfully.",
        "doc_id": doc_id,
        "chunks_indexed": count,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        crew = get_crew()
        result = crew.run(question)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to generate answer: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
