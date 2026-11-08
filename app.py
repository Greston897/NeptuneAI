from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from classifier import Classifier, MAX_TEXT_CHARS

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifacts/rubert-tiny2-toxicity"))
app = FastAPI(title="NeptuneAI moderation", docs_url="/docs", redoc_url=None)
classifier: Classifier | None = None
load_error: str | None = None


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]


@app.on_event("startup")
def load_model() -> None:
    global classifier, load_error
    try:
        classifier = Classifier(MODEL_DIR)
        load_error = None
    except Exception as exc:
        # Keep the health endpoint and UI available; classification must fail closed
        # with 503 rather than silently pretending to have evaluated the text.
        classifier = None
        load_error = str(exc)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready" if classifier is not None else "model_not_loaded"}


@app.post("/api/classify", response_class=PlainTextResponse)
def classify(request: ClassifyRequest) -> PlainTextResponse:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не обучена или не загружена")
    try:
        return PlainTextResponse(classifier.classify(request.text))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
