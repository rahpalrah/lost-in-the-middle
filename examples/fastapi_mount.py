"""Mount the longdoc API inside an existing FastAPI app, behind your own auth.

    pip install "longdoc-qa[server]"
    uvicorn examples.fastapi_mount:app --reload

Routes appear under /longdoc/v1/... and in your app's OpenAPI docs at /docs.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException

from longdoc import DocumentService, JSONFileStore, QAConfig
from longdoc.server import create_router


def require_user(x_user_id: str = Header()) -> str:
    """Stand-in for your existing authentication dependency."""
    if not x_user_id:
        raise HTTPException(status_code=401)
    return x_user_id


service = DocumentService(
    store=JSONFileStore("./longdoc-data"),
    config=QAConfig(effort="medium", max_parents=4),
)

app = FastAPI(title="My existing API")
app.include_router(
    create_router(service),  # no api_key: your own dependency below authenticates
    prefix="/longdoc",
    dependencies=[Depends(require_user)],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"hello": "world"}
