"""Add document Q&A to an existing Flask app (or any WSGI framework).

    pip install "longdoc-qa[pdf]" flask
    FLASK_APP=examples/flask_integration.py flask run

`DocumentService` uses only the standard library by default: no extra packages or API keys
needed to get started. Set ANTHROPIC_API_KEY to use Claude; without it, the built-in
extractive engine answers by quoting the document. Create one service per process and call
it from your own routes, behind your own authentication.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

from longdoc import (
    DocumentNotFoundError,
    DocumentService,
    JSONFileStore,
    LongDocError,
    RefusalError,
    answer_to_dict,
)

app = Flask(__name__)
service = DocumentService(store=JSONFileStore("./longdoc-data"))


@app.post("/contracts/<contract_id>/document")
def upload(contract_id: str) -> Any:
    file = request.files["file"]
    # Reuse your own primary key as the document ID so the two stay linked.
    info = service.add_bytes(
        file.read(), filename=file.filename or "upload", document_id=contract_id
    )
    return jsonify(info.to_dict()), 201


@app.post("/contracts/<contract_id>/questions")
def ask(contract_id: str) -> Any:
    question = (request.get_json(silent=True) or {}).get("question", "")
    answer = service.ask(contract_id, question)
    return jsonify(answer_to_dict(answer))


@app.errorhandler(DocumentNotFoundError)
def not_found(exc: DocumentNotFoundError) -> Any:
    return jsonify(error="not_found", message=str(exc)), 404


@app.errorhandler(RefusalError)
def refused(exc: RefusalError) -> Any:
    return jsonify(error="refused", message=str(exc), category=exc.category), 422


@app.errorhandler(LongDocError)
def bad_request(exc: LongDocError) -> Any:
    return jsonify(error="bad_request", message=str(exc)), 400
