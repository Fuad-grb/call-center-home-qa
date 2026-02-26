"""
FastAPI REST endpoint for the QA pipeline.

POST /evaluate — evaluate a single call transcript
POST /evaluate/batch — evaluate multiple transcripts
GET /health — health check
"""

from __future__ import annotations
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.logger import setup_logging, get_logger
from app.pipeline import evaluate_call

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Kontakt Home QA Pipeline",
    description="Call-center keyfiyyətə nəzarət sistemi",
    version="1.0.0",
)


# ── Request/Response models ──

class EvaluateRequest(BaseModel):
    call_id: str
    segments: list[dict[str, Any]]


class BatchRequest(BaseModel):
    calls: list[dict[str, Any]]


# ── Endpoints ──

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    """Evaluate a single call transcript."""
    logger.info("API request", extra={"call_id": request.call_id})
    try:
        result = evaluate_call(request.model_dump())
        return result.model_dump()
    except Exception as e:
        logger.error("Evaluation failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/batch")
def evaluate_batch(request: BatchRequest):
    """Evaluate multiple call transcripts."""
    results = []
    for i, call_data in enumerate(request.calls):
        logger.info(f"Batch {i+1}/{len(request.calls)}")
        try:
            result = evaluate_call(call_data)
            results.append(result.model_dump())
        except Exception as e:
            results.append({"call_id": call_data.get("call_id", "?"), "error": str(e)})
    return results
