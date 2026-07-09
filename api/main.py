import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Import pipeline and PDF extractor
from graph import placement_workflow
from agents.placement_agents import extract_text_from_pdf

# ─────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate pipeline compiles cleanly on startup."""
    print("[OK] HireReady AI - LangGraph pipeline loaded successfully.")
    yield

app = FastAPI(
    title="HireReady AI — Placement Evaluation API",
    description=(
        "Multi-agent LangGraph pipeline powered by Gemini (gemini-2.0-flash). "
        "Runs 6 specialised agents in sequence — Resume Review → ATS Optimisation → "
        "Skill Gap Analysis → Study Planning → Mock Interview → GD Simulation — "
        "with a critic node that loops back when readiness < 70 (up to 3 iterations)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow Streamlit UI and local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """
    Lightweight health-check endpoint used by Render and load balancers.
    Returns 200 when the API is ready to accept requests.
    """
    return {
        "status": "healthy",
        "service": "HireReady AI Placement API",
        "version": "2.0.0",
        "pipeline_nodes": [
            "resume_reviewer",
            "ats_optimizer",
            "skill_gap_analyzer",
            "study_planner",
            "interviewer",
            "gd_moderator",
            "critic",
        ],
    }


# ─────────────────────────────────────────────
# POST /analyze
# ─────────────────────────────────────────────
@app.post("/analyze", tags=["Pipeline"])
async def analyze_candidate(
    job_description: str = Form(
        ...,
        description="Full text of the target job description.",
        examples=["Role: Backend Engineer\nRequirements: FastAPI, PostgreSQL, AWS..."],
    ),
    file: UploadFile = File(
        ...,
        description="Candidate resume in PDF format.",
    ),
):
    """
    Accepts a resume PDF and job description, extracts resume text via pdfplumber,
    then runs the full 6-agent LangGraph pipeline with the critic feedback loop.

    Returns the complete **CandidateState** as JSON, including:
    - `resume_review` — structured resume feedback
    - `resume_text` — ATS-optimised resume rewrite
    - `ats_score` — keyword compatibility score (0–100)
    - `skill_gaps` — list of missing competencies
    - `study_plan` — week-by-week learning curriculum
    - `interview_feedback` — 5 tailored mock Q&As
    - `gd_feedback` — GD topic simulation + evaluation
    - `readiness_score` — final placement readiness score (0–100)
    - `iteration_count` — number of pipeline passes completed
    """
    # ── Validate file type ──────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf resume.",
        )

    # ── Extract text from PDF ───────────────────────────────────
    tmp_path: str | None = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        resume_text = extract_text_from_pdf(tmp_path)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not resume_text.strip() or resume_text.startswith("Error"):
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse text from PDF: {resume_text}",
        )

    # ── Build initial CandidateState ───────────────────────────
    initial_state = {
        "resume_text": resume_text,
        "job_description": job_description,
        "ats_score": None,
        "skill_gaps": [],
        "study_plan": None,
        "interview_feedback": None,
        "gd_feedback": None,
        "readiness_score": None,
        "iteration_count": 0,
        "resume_review": None,
    }

    # ── Run the LangGraph pipeline ─────────────────────────────
    try:
        result_state = placement_workflow.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(exc)}",
        )

    return result_state


# ─────────────────────────────────────────────
# Local dev entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
