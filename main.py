"""
backend/main.py

Minimal FastAPI backend. For now it just serves the pre-generated season
trace (backend/data/sample_trace.json) as JSON. Later, /api/trace can be
upgraded to run the ACTUAL trained agents instead of reading a static file --
the response shape stays identical, so the frontend never has to change.

Run this from inside the backend/ folder:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs in a browser -- FastAPI auto-generates
an interactive test page there, so you can check the API works BEFORE the
frontend is involved at all. That's the easiest way to debug backend issues.
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Irrigation API")

# CORS: without this, a browser blocks the frontend (opened as a file, or
# served on a different port like 5500) from calling this API on port 8000.
# This is the #1 thing beginners get stuck on -- if your frontend fetch()
# silently fails with a network error in the browser console, check this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"


@app.get("/")
def root():
    return {"status": "ok", "message": "Irrigation API running. See /docs or /api/trace"}


@app.get("/api/trace")
def get_trace(policy: str = "threshold"):
    """
    Serves a pre-generated season trace. ?policy= one of: random, fixed,
    threshold (and later, once Phase 4 is done: trained).
    Generate these files first by running, from the project root:
        python generate_trace.py fixed
        python generate_trace.py threshold
        python generate_trace.py random
    """
    file_path = DATA_DIR / f"trace_{policy}.json"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trace file for policy '{policy}'. Run: python generate_trace.py {policy}",
        )
    with open(file_path) as f:
        return json.load(f)