"""
main.py — FastAPI server exposing the pipeline as HTTP endpoints.
This is what the frontend (HTML/CSS/JS) will call:
  POST   /ask                       - ask a question, get back answer + citations + verdict
  POST   /upload                    - upload a document into data/incoming/ (watcher picks it up)
  GET    /health                    - simple check that the server is alive
  GET    /kg/summary                - quick knowledge graph stats, useful for a demo dashboard
  GET    /sessions                  - list all chat session IDs
  GET    /sessions/{id}/history     - full Q&A history for one session
  DELETE /sessions/{id}             - delete a chat session's history
"""

from fastapi import Header, HTTPException
from src import auth
from fastapi import Depends
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src import memory
from src import config
from src import pipeline
from src import kg_retriever

def get_current_user(x_username: str = Header(default="guest"), x_role: str = Header(default="technician")) -> auth.User:
    """
    Reads identity from request headers. This is a lightweight stand-in
    for real authentication (no login/session yet) — good enough to
    demonstrate RBAC enforcement without building a full auth system.
    """
    try:
        role = auth.Role(x_role.lower())
    except ValueError:
        role = auth.Role.TECHNICIAN
    return auth.User(username=x_username, role=role)

app = FastAPI(title="Industrial Knowledge Assistant API")

# Allows the frontend (a separate HTML file, possibly opened straight from disk
# or served on a different port) to actually call this API from the browser.
# CORS blocks cross-origin requests by default — this opens it up for local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    """
    Runs the full pipeline: hybrid retrieval -> reasoning -> verification
    -> citation, and returns the complete result as JSON.
    """
    result = pipeline.ask(request.query, session_id=request.session_id)
    return result

@app.get("/sessions")
def list_sessions():
    return {"sessions": memory.list_sessions()}


@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str):
    return {"session_id": session_id, "history": memory.get_history(session_id)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Deletes a chat session's saved history (used by the sidebar delete button)."""
    memory.clear_session(session_id)
    return {"session_id": session_id, "status": "deleted"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), user: auth.User = Depends(get_current_user)):
    if not user.can("upload_documents"):
        raise HTTPException(status_code=403, detail=f"User '{user.username}' ({user.role.value}) cannot upload documents")

    config.INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.INCOMING_DIR / file.filename
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    return {"filename": file.filename, "status": "saved, awaiting processing by watcher"}


@app.get("/kg/summary")
def kg_summary():
    """Quick knowledge graph stats — useful for a demo/dashboard view."""
    return kg_retriever.graph_summary()