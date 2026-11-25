"""FitBite AI - FastAPI backend.

Run from the project root (the folder that contains the `app/` package):
    python -m uvicorn server:app --reload
Then open http://127.0.0.1:8000
"""
import logging
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import NutritionAgent

logger = logging.getLogger("fitbite")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_SESSIONS = 100

app = FastAPI(title="FitBite AI")

# One agent per browser session, so conversation state is not shared between users.
# If your NutritionAgent is stateless (or heavy to create), replace this with a
# single shared instance: `agent = NutritionAgent()`.
_agents: "OrderedDict[str, NutritionAgent]" = OrderedDict()
_lock = threading.Lock()


def get_agent(session_id: str) -> NutritionAgent:
    with _lock:
        if session_id in _agents:
            _agents.move_to_end(session_id)
            return _agents[session_id]
        agent = NutritionAgent()
        _agents[session_id] = agent
        while len(_agents) > MAX_SESSIONS:
            _agents.popitem(last=False)
        return agent


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=100)


class ChatResponse(BaseModel):
    answer: str


# Plain `def` endpoints run in FastAPI's threadpool, so a slow, blocking
# agent.chat() call does not freeze the server.
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")
    try:
        answer = get_agent(req.session_id).chat(message)
    except Exception as exc:
        logger.exception("Agent failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return ChatResponse(answer=str(answer))


@app.post("/api/reset")
def reset(req: ResetRequest):
    with _lock:
        _agents.pop(req.session_id, None)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Missing {page}. Put index.html inside the 'static' folder next to server.py.",
        )
    return FileResponse(page)


STATIC_DIR.mkdir(exist_ok=True)  # prevents a startup crash if the folder is missing
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
