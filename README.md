# HireReady AI — Multi-Agent Placement Preparation System

> **An agentic LangGraph pipeline powered by Google Gemini that reviews resumes, closes skill gaps, and prepares candidates end-to-end for technical placement interviews.**

---

## 📌 What It Does

HireReady AI runs a candidate's resume and a target job description through a **6-agent LangGraph workflow** with an intelligent critic feedback loop:

```
START
  │
  ▼
resume_reviewer      ← Analyses resume structure, formatting & role alignment
  │
  ▼
ats_optimizer        ← Rewrites the resume for maximum ATS keyword compatibility
  │
  ▼
skill_gap_analyzer   ← Identifies missing skills vs. the JD; calculates ATS score
  │
  ▼
study_planner        ← Generates a week-by-week curriculum to close skill gaps
  │
  ▼
interviewer          ← Creates 5 tailored mock interview Q&As with answer guides
  │
  ▼
gd_moderator         ← Simulates a Group Discussion transcript + moderator evaluation
  │
  ▼
critic               ← Calculates readiness_score
  │
  ├─ readiness < 70 AND iterations < 3  ──► loop back to resume_reviewer
  │
  └─ otherwise ──────────────────────────► END
```

Each agent receives and returns the full **`CandidateState`** TypedDict, keeping all intermediate outputs available to downstream agents.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini via `langchain-google-genai` |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| **API Backend** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **PDF Parsing** | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| **Deployment** | [Render](https://render.com/) via Docker |
| **Config** | `python-dotenv` + `.env` file |

---

## 📁 Project Structure

```
HireReady AI/
├── agents/
│   ├── __init__.py            # Exports all node functions
│   ├── resume_reviewer.py     # Agent 1 — Resume review & feedback
│   ├── ats_optimizer.py       # Agent 2 — ATS-optimised rewrite
│   ├── skill_gap_analyzer.py  # Agent 3 — Skill gap detection + ATS score
│   ├── study_planner.py       # Agent 4 — Week-by-week study plan
│   ├── interviewer.py         # Agent 5 — 5 mock interview Q&As
│   ├── gd_moderator.py        # Agent 6 — GD simulation + evaluation
│   └── placement_agents.py    # Legacy utility class (PDF extraction etc.)
│
├── graph/
│   ├── __init__.py            # Exports CandidateState + placement_workflow
│   ├── state.py               # CandidateState TypedDict definition
│   ├── pipeline.py            # Full 6-agent StateGraph with critic loop  ← MAIN
│   └── workflow.py            # Legacy workflow (superseded by pipeline.py)
│
├── api/
│   └── main.py                # FastAPI app — POST /analyze, GET /health
│
├── ui/
│   └── app.py                 # Streamlit dashboard
│
├── Dockerfile                 # Production container for Render
├── render.yaml                # Render deployment configuration
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variable template
```

---

## ⚙️ Local Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-org/hireready-ai.git
cd hireready-ai
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:

```dotenv
GOOGLE_API_KEY=AIza...
HOST=127.0.0.1
PORT=8000
```

> **No API key?** The pipeline runs on high-fidelity mock data automatically when the key is absent or left as the placeholder value.

### 3. Run the FastAPI backend

```bash
# From the project root
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs will be available at **http://127.0.0.1:8000/docs**

### 4. Run the Streamlit UI

Open a second terminal:

```bash
streamlit run ui/app.py
```

The dashboard opens at **http://localhost:8501**

### 5. Test the pipeline directly (no UI)

```python
from dotenv import load_dotenv
load_dotenv()

from graph import placement_workflow

state = placement_workflow.invoke({
    "resume_text": "Jane Doe — Python developer with 2 years FastAPI experience...",
    "job_description": "Role: Backend Engineer. Requires: FastAPI, PostgreSQL, Docker, AWS.",
    "ats_score": None,
    "skill_gaps": [],
    "study_plan": None,
    "interview_feedback": None,
    "gd_feedback": None,
    "readiness_score": None,
    "iteration_count": 0,
    "resume_review": None,
})

print("Readiness score:", state["readiness_score"])
print("Pipeline passes:", state["iteration_count"])
```

---

## 🌐 API Reference

### `GET /health`
Returns service status and registered pipeline nodes.

```json
{
  "status": "healthy",
  "service": "HireReady AI Placement API",
  "version": "2.0.0",
  "pipeline_nodes": ["resume_reviewer", "ats_optimizer", "..."]
}
```

### `POST /analyze`
Accepts a **PDF resume** and **job description** (multipart form data). Runs the full pipeline and returns the complete `CandidateState` as JSON.

**Request (form-data):**

| Field | Type | Description |
|---|---|---|
| `file` | `File` | Candidate resume (PDF) |
| `job_description` | `string` | Full text of the target JD |

**Example (curl):**

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "job_description=Role: Backend Engineer. Requires FastAPI, Docker." \
  -F "file=@/path/to/resume.pdf"
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `resume_review` | `string` | Structured resume feedback from Agent 1 |
| `resume_text` | `string` | ATS-optimised resume rewrite from Agent 2 |
| `ats_score` | `float` | Keyword match score 0–100 |
| `skill_gaps` | `list[str]` | Missing skills/technologies |
| `study_plan` | `string` | Week-by-week study curriculum |
| `interview_feedback` | `string` | 5 mock interview Q&As with answer guides |
| `gd_feedback` | `string` | GD topic simulation + moderator evaluation |
| `readiness_score` | `float` | Final placement readiness score 0–100 |
| `iteration_count` | `int` | Number of pipeline passes completed |

---

## 🚀 Deploy on Render

### Prerequisites
- A [Render](https://render.com/) account
- The project pushed to a GitHub repository

### Steps

1. **Create a new Render project** and connect your GitHub repo.

2. **Set environment variables** in the Render dashboard under **Environment**:
   ```
   GOOGLE_API_KEY=AIza...
   ```

3. **Render auto-detects** the `render.yaml` and `Dockerfile` in the project root and builds automatically.

4. The **health check** at `/health` is polled by Render to confirm successful deployment.

5. Once deployed, copy your Render **public URL** and set it in the Streamlit UI:
   ```dotenv
   API_URL=https://your-app.onrender.com
   ```
   Or pass it as an env variable in a separate Streamlit service on Render.

### Build locally with Docker

```bash
# Build image
docker build -t hireready-ai .

# Run container
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=AIza... \
  hireready-ai
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Optional | *(mock mode)* | Google API key for live Gemini inference |
| `HOST` | No | `127.0.0.1` | Uvicorn bind host (set to `0.0.0.0` in prod) |
| `PORT` | No | `8000` | Uvicorn listen port (injected by Render automatically) |
| `API_URL` | No | `http://127.0.0.1:8000` | FastAPI base URL used by the Streamlit UI |

---

## 📄 License

MIT — feel free to fork, extend, and deploy.
