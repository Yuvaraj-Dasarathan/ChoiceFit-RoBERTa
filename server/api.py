"""FastAPI service for the ChoiceFit model.

Run:  .venv/bin/uvicorn server.api:app --host 127.0.0.1 --port 8000
Serves POST /predict, GET /health, and the static page in web/ (if present).
"""
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.model import DEFAULT_QUESTION, ChoiceModel

ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL = ROOT / "artifacts/sagemaker/choicefit-l4-full-b8-001"
MODEL_ID = os.getenv("CHOICEFIT_MODEL", str(LOCAL_MODEL))
ALLOWED_ORIGINS = [o for o in os.getenv("CHOICEFIT_ORIGINS", "").split(",") if o]
RATE_LIMIT = int(os.getenv("CHOICEFIT_RATE_PER_MIN", "20"))  # per client IP

MAX_OPTIONS = 8
MAX_CHARS = 2000
MAX_OPTION_CHARS = 300

app = FastAPI(title="ChoiceFit API")
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

_model = None
_lock = threading.Lock()  # one inference at a time keeps CPU/memory bounded
_hits = defaultdict(deque)


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHARS)
    question: str = Field(default=DEFAULT_QUESTION, max_length=500)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)


def client_ip(request: Request) -> str:
    # The last X-Forwarded-For entry is the one ngrok itself observed
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def check_rate(ip: str) -> None:
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT} requests per minute.")
    window.append(now)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    global _model
    check_rate(client_ip(request))
    options = [o.strip() for o in body.options]
    if any(not o for o in options):
        raise HTTPException(422, "Options must not be empty.")
    if any(len(o) > MAX_OPTION_CHARS for o in options):
        raise HTTPException(422, f"Each option must be under {MAX_OPTION_CHARS} characters.")
    if len(set(options)) != len(options):
        raise HTTPException(422, "Options must be different from each other.")
    question = body.question.strip() or DEFAULT_QUESTION
    choices = {f"option_{i}": text for i, text in enumerate(options)}
    with _lock:
        if _model is None:
            _model = ChoiceModel(MODEL_ID, device="cpu")
        result = _model.predict(body.message.strip(), choices, question, pair_batch_size=8)
    probabilities = [result["probabilities"][f"option_{i}"] for i in range(len(options))]
    best = max(range(len(options)), key=probabilities.__getitem__)
    return {
        "best_index": best,
        "best_option": options[best],
        "results": [{"option": o, "probability": p} for o, p in zip(options, probabilities)],
    }


WEB_DIR = ROOT / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
