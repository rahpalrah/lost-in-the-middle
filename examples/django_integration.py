"""Use longdoc from a Django view.

Put the service in a module that is imported once per process (for example
``yourapp/services.py``), then call it from views, Celery tasks or management commands.
"""

from __future__ import annotations

import json
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_POST

from longdoc import DocumentNotFoundError, DocumentService, JSONFileStore, answer_to_dict

# yourapp/services.py
longdoc_service = DocumentService(store=JSONFileStore("/var/lib/yourapp/longdoc"))


# yourapp/views.py
@require_POST
def ask_contract(request: HttpRequest, contract_id: str) -> Any:
    question = json.loads(request.body or b"{}").get("question", "")
    try:
        answer = longdoc_service.ask(contract_id, question)
    except DocumentNotFoundError:
        return JsonResponse({"error": "not_found"}, status=404)
    return JsonResponse(answer_to_dict(answer))


# yourapp/tasks.py (Celery): index uploads in the background, not in the request.
def index_contract(contract_id: str, path: str) -> None:
    longdoc_service.add_file(path, document_id=contract_id)
