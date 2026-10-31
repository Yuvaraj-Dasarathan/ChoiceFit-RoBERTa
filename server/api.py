"""FastAPI service for the ChoiceFit model.

Run:  .venv/bin/uvicorn server.api:app --host 127.0.0.1 --port 8000
Serves POST /predict, GET /health, and the static page in web/ (if present).
"""
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
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

def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


IP_PER_MIN = _int("CHOICEFIT_RATE_PER_MIN", 20)       # per client IP
IP_PER_DAY = _int("CHOICEFIT_IP_PER_DAY", 200)        # per client IP
GLOBAL_PER_MIN = _int("CHOICEFIT_GLOBAL_PER_MIN", 60)  # all clients together
GLOBAL_PER_DAY = _int("CHOICEFIT_GLOBAL_PER_DAY", 2000)
MAX_WAITING = _int("CHOICEFIT_MAX_QUEUE", 5)          # requests running or waiting
DISABLED = os.getenv("CHOICEFIT_DISABLED") == "1"     # kill switch

LOG_FILE = Path(os.getenv("CHOICEFIT_LOG", str(ROOT / "logs/requests.jsonl")))
_log_lock = threading.Lock()
LOGGED_HEADERS = ("user-agent", "accept-language", "referer", "origin", "sec-ch-ua", "sec-ch-ua-platform",
                  "sec-ch-ua-mobile", "x-forwarded-for", "x-forwarded-proto", "cf-ipcountry")

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
_state = threading.Lock()  # guards the counters below
_ip_min = defaultdict(deque)
_ip_day = defaultdict(deque)
_global_min = deque()
_global_day = deque()
_in_flight = 0


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHARS)
    question: str = Field(default=DEFAULT_QUESTION, max_length=500)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)
    client: dict[str, str] | None = Field(default=None, max_length=10)


def client_ip(request: Request) -> str:
    # ngrok appends the real client IP last; earlier entries can be spoofed by the caller.
    # Headers can arrive as several lines, so look at every line, not just the first.
    entries = [p.strip() for line in request.headers.getlist("x-forwarded-for") for p in line.split(",")]
    entries = [e for e in entries if e]
    if entries:
        return entries[-1]
    return request.client.host if request.client else "unknown"


def log_request(request: Request, ip: str, status: int, entry: dict) -> None:
    """Append one JSON line per request. Logging must never break a prediction."""
    try:
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ip": ip,
            "status": status,
            "headers": {h: request.headers.get(h) for h in LOGGED_HEADERS if request.headers.get(h)},
            **entry,
        }
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _prune(window: deque, now: float, span: float) -> None:
    while window and now - window[0] > span:
        window.popleft()


def admit(ip: str) -> None:
    """Raise 429/503 if a limit is hit; otherwise count the request and reserve a queue slot."""
    global _in_flight
    now = time.monotonic()
    with _state:
        if DISABLED:
            raise HTTPException(503, "The demo is switched off.")
        for window, span, limit, msg in (
            (_ip_min[ip], 60, IP_PER_MIN, f"Too many requests: {IP_PER_MIN} per minute per visitor."),
            (_ip_day[ip], 86400, IP_PER_DAY, f"Daily limit reached: {IP_PER_DAY} requests per visitor."),
            (_global_min, 60, GLOBAL_PER_MIN, "The demo is busy. Please try again in a minute."),
            (_global_day, 86400, GLOBAL_PER_DAY, "The demo reached its daily limit. Please try again tomorrow."),
        ):
            _prune(window, now, span)
            if len(window) >= limit:
                raise HTTPException(429, msg)
        if _in_flight >= MAX_WAITING:
            raise HTTPException(503, "The demo is busy. Please try again in a few seconds.")
        for window in (_ip_min[ip], _ip_day[ip], _global_min, _global_day):
            window.append(now)
        _in_flight += 1
        if len(_ip_day) > 5000:  # drop idle visitors so memory stays bounded
            for key in [k for k, w in _ip_day.items() if not w or now - w[-1] > 86400]:
                _ip_day.pop(key, None)
                _ip_min.pop(key, None)


def release() -> None:
    global _in_flight
    with _state:
        _in_flight -= 1


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    ip = client_ip(request)
    started = time.monotonic()
    entry = {"message": body.message, "question": body.question, "options": body.options,
             "client": {k[:40]: v[:200] for k, v in (body.client or {}).items()}}
    try:
        response = _predict(body, ip)
    except HTTPException as e:
        log_request(request, ip, e.status_code, {**entry, "error": e.detail})
        raise
    except Exception as e:
        log_request(request, ip, 500, {**entry, "error": repr(e)})
        raise
    log_request(request, ip, 200, {**entry, "result": response, "seconds": round(time.monotonic() - started, 3)})
    return response


def _predict(body: PredictRequest, ip: str):
    global _model
    options = [o.strip() for o in body.options]
    if any(not o for o in options):
        raise HTTPException(422, "Options must not be empty.")
    if any(len(o) > MAX_OPTION_CHARS for o in options):
        raise HTTPException(422, f"Each option must be under {MAX_OPTION_CHARS} characters.")
    if len(set(options)) != len(options):
        raise HTTPException(422, "Options must be different from each other.")
    question = body.question.strip() or DEFAULT_QUESTION
    choices = {f"option_{i}": text for i, text in enumerate(options)}
    admit(ip)
    try:
        with _lock:
            if _model is None:
                _model = ChoiceModel(MODEL_ID, device="cpu")
            result = _model.predict(body.message.strip(), choices, question, pair_batch_size=8)
    finally:
        release()
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
