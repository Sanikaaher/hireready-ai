import os
from graph.state import CandidateState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your-"):
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.5
    )

_MOCK_QA = """## 🎯 Mock Interview — 5 Tailored Q&As

---

### Q1. Tell me about a backend project where you owned the full development lifecycle.
**💡 Ideal Answer Guide:**
Open with the project context (what problem it solved), walk through your technical decisions
(language, framework, database choice), highlight a challenge you overcame (e.g., scaling a bottleneck,
fixing a security issue), and close with measurable outcomes. Use the STAR format: Situation → Task → Action → Result.

---

### Q2. How would you design a RESTful API for a high-traffic application using FastAPI?
**💡 Ideal Answer Guide:**
Discuss async route handlers (`async def`), dependency injection for DB sessions, Pydantic models
for request/response validation, and middleware for rate limiting and auth. Mention horizontal scaling
with Gunicorn + Uvicorn workers, caching hot routes with Redis, and structured logging with
correlation IDs. Bonus: mention background tasks with `BackgroundTasks` or Celery.

---

### Q3. Describe a time you had to debug a production issue under time pressure.
**💡 Ideal Answer Guide:**
Describe the incident clearly (e.g., API latency spike, failed deployment), your triage process
(checking logs, metrics dashboards, isolating the service), the root cause, and the fix you shipped.
Emphasize communication with stakeholders and any post-mortem actions taken to prevent recurrence.

---

### Q4. How do you ensure data integrity when working with relational databases and ORM layers?
**💡 Ideal Answer Guide:**
Mention ACID transactions, proper use of database constraints (unique, FK, NOT NULL), ORM-level
validators, and schema migration strategies (Alembic for SQLAlchemy). Discuss soft deletes vs hard
deletes, handling race conditions with `SELECT FOR UPDATE`, and writing integration tests that cover
DB rollback scenarios.

---

### Q5. Where do you see AI/LLM tooling fitting into modern backend engineering workflows?
**💡 Ideal Answer Guide:**
Reference concrete use cases: AI-assisted code review, LLM-powered data extraction pipelines,
semantic search with vector DBs (pgvector, Pinecone), or agentic workflows (LangGraph, AutoGen).
Demonstrate awareness of production concerns: token cost management, prompt injection risks,
hallucination handling, and observability with tools like LangSmith or Phoenix.
"""

def interviewer_node(state: CandidateState) -> CandidateState:
    """
    Generates 5 targeted mock interview Q&As based on the candidate's resume
    and job description using Gemini API.
    Receives and returns CandidateState.
    """
    print("--- RUNNING INTERVIEWER AGENT NODE ---")
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")

    llm = get_llm()
    if not llm:
        feedback = _MOCK_QA
    else:
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are a Senior Technical Interviewer at a top tech company.\n"
                "Based on the candidate's resume and the target job description below, "
                "generate exactly **5 mock interview questions** that are highly specific "
                "to this candidate's background and role.\n\n"
                "For each question:\n"
                "  - Number it clearly (Q1 through Q5).\n"
                "  - Mix question types: at least 2 behavioral (STAR-format), "
                "2 technical deep-dives, and 1 culture/growth question.\n"
                "  - Follow each question with a detailed **'Ideal Answer Guide'** (3–5 sentences) "
                "describing what a strong candidate answer should include.\n\n"
                "Resume:\n{resume_text}\n\n"
                "Job Description:\n{job_description}\n\n"
                "Format output in clean, readable Markdown with separators between questions."
            )
            chain = prompt | llm
            response = chain.invoke({
                "resume_text": resume_text,
                "job_description": job_description
            })
            feedback = response.content
        except Exception as e:
            feedback = _MOCK_QA
            feedback += f"\n\n> ⚠️ *Live generation error (using detailed fallback): {str(e)}*"

    updated_state = dict(state)
    updated_state["interview_feedback"] = feedback
    return updated_state
