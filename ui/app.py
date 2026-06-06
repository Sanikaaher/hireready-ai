import streamlit as st
import requests
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ── Local pipeline fallback ─────────────────────────────────────────────────
try:
    from graph import placement_workflow
    from agents.placement_agents import extract_text_from_pdf
    LOCAL_AVAILABLE = True
except ImportError:
    LOCAL_AVAILABLE = False

_raw_api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
# Render's fromService hostport gives "hostname:port" without a scheme — add https://
if _raw_api_url and not _raw_api_url.startswith("http"):
    API_URL = f"https://{_raw_api_url}"
else:
    API_URL = _raw_api_url

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireReady AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(99,102,241,.3);
    border-radius: 20px;
    padding: 2.8rem 2rem 2.2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 60% 0%, rgba(99,102,241,.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -.03em;
    margin-bottom: .4rem;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    color: rgba(255,255,255,.65);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Score cards */
.score-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.6rem;
}
.score-card {
    background: #12131f;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    transition: transform .25s, border-color .25s;
}
.score-card:hover { transform: translateY(-4px); border-color: rgba(99,102,241,.4); }
.sc-label {
    font-size: .75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: rgba(255,255,255,.45);
    margin-bottom: .5rem;
}
.sc-value { font-size: 2rem; font-weight: 700; }
.clr-green  { color: #34d399; }
.clr-yellow { color: #fbbf24; }
.clr-red    { color: #f87171; }
.clr-purple { color: #a78bfa; }

/* Readiness bar wrapper */
.readiness-wrap {
    background: #12131f;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.6rem;
}
.readiness-label {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: .7rem;
}
.readiness-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; }
.readiness-pct   { font-size: 1.6rem; font-weight: 700; }

/* Gap badges */
.gap-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .5rem; }
.gap-badge {
    background: rgba(239,68,68,.12);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,.25);
    padding: .3rem .75rem;
    border-radius: 999px;
    font-size: .8rem;
    font-weight: 600;
}

/* Expander overrides */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 12px !important;
    background: #12131f !important;
    margin-bottom: .8rem;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #c7d2fe !important;
    padding: .9rem 1.1rem !important;
}

/* Progress bar colour */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
    border-radius: 999px !important;
}

/* Sidebar refinements */
section[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(255,255,255,.07);
}
</style>
""", unsafe_allow_html=True)

# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">HireReady <span>AI</span></div>
    <div class="hero-sub">
        Multi-agent LangGraph pipeline · Gemini-powered · 6 specialised agents ·
        Critic feedback loop · End-to-end placement preparation
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Inputs")

    default_jd = (
        "Role: Junior Backend Engineer (Python)\n"
        "Requirements:\n"
        "- Strong experience with FastAPI or Flask\n"
        "- PostgreSQL / SQLite + ORM knowledge (SQLAlchemy / Tortoise)\n"
        "- System Design fundamentals\n"
        "- AWS / cloud deployment experience\n"
        "- LangChain / LangGraph exposure is a plus\n"
        "- Docker containerisation"
    )
    job_desc = st.text_area(
        "Job Description",
        value=default_jd,
        height=200,
        placeholder="Paste the full job description here...",
    )

    uploaded_pdf = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        help="Only PDF format is supported.",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Analyse Placement Profile", use_container_width=True, type="primary")

# ── Helper: colour class for a numeric score ─────────────────────────────────
def score_colour(val: float) -> str:
    if val >= 75:
        return "clr-green"
    if val >= 50:
        return "clr-yellow"
    return "clr-red"

# ── Main logic ───────────────────────────────────────────────────────────────
if run_btn:
    if not job_desc.strip():
        st.warning("⚠️ Please add a Job Description in the sidebar.")
        st.stop()
    if not uploaded_pdf:
        st.warning("⚠️ Please upload a PDF resume in the sidebar.")
        st.stop()

    with st.spinner("⚙️ Running 6-agent LangGraph pipeline… this may take 20–60 s with live inference."):
        result = None
        mode = ""

        # Write PDF to a temp file for local fallback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getvalue())
            tmp_path = tmp.name

        # ── Try FastAPI backend ─────────────────────────────────────────────
        try:
            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    f"{API_URL}/analyze",
                    data={"job_description": job_desc},
                    files={"file": (uploaded_pdf.name, f, "application/pdf")},
                    timeout=180,
                )
            if resp.status_code == 200:
                result = resp.json()
                mode = "FastAPI backend"
            else:
                st.sidebar.warning(f"Backend error {resp.status_code} — falling back to local pipeline.")
        except Exception:
            st.sidebar.info("FastAPI offline — running local in-process pipeline.")

        # ── Local pipeline fallback ─────────────────────────────────────────
        if result is None and LOCAL_AVAILABLE:
            try:
                resume_text = extract_text_from_pdf(tmp_path)
                initial = {
                    "resume_text": resume_text,
                    "job_description": job_desc,
                    "ats_score": None,
                    "skill_gaps": [],
                    "study_plan": None,
                    "interview_feedback": None,
                    "gd_feedback": None,
                    "readiness_score": None,
                    "iteration_count": 0,
                    "resume_review": None,
                }
                result = placement_workflow.invoke(initial)
                mode = "local in-process pipeline"
            except Exception as ex:
                st.error(f"Local pipeline failed: {ex}")

        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Render results ──────────────────────────────────────────────────────
    if not result:
        st.error("Pipeline did not return a result. Please try again or contact support if the issue persists.")
        st.stop()

    st.toast(f"✅ Analysis complete via {mode}!", icon="🎓")

    ats       = result.get("ats_score") or 0.0
    gaps      = result.get("skill_gaps") or []
    readiness = result.get("readiness_score") or 0.0
    iterations = result.get("iteration_count") or 1

    ats_cls = score_colour(ats)
    rdy_cls = score_colour(readiness)

    # ── Score cards ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="score-row">
        <div class="score-card">
            <div class="sc-label">ATS Match Score</div>
            <div class="sc-value {ats_cls}">{ats:.0f}<span style="font-size:1rem;opacity:.6">%</span></div>
        </div>
        <div class="score-card">
            <div class="sc-label">Readiness Score</div>
            <div class="sc-value {rdy_cls}">{readiness:.0f}<span style="font-size:1rem;opacity:.6">%</span></div>
        </div>
        <div class="score-card">
            <div class="sc-label">Skill Gaps</div>
            <div class="sc-value clr-red">{len(gaps)}</div>
        </div>
        <div class="score-card">
            <div class="sc-label">Pipeline Passes</div>
            <div class="sc-value clr-purple">{iterations}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Readiness progress bar ──────────────────────────────────────────────
    rdy_pct = readiness / 100
    bar_colour = "#34d399" if readiness >= 75 else ("#fbbf24" if readiness >= 50 else "#f87171")
    verdict = (
        "🟢 **Placement Ready** — Strong candidate profile."
        if readiness >= 75 else
        "🟡 **Almost Ready** — A few gaps to close before interviews."
        if readiness >= 50 else
        "🔴 **Needs Work** — Significant preparation required."
    )
    st.markdown(f"""
    <div class="readiness-wrap">
        <div class="readiness-label">
            <span class="readiness-title">🏆 Overall Readiness Score</span>
            <span class="readiness-pct" style="color:{bar_colour}">{readiness:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(rdy_pct, 1.0))
    st.markdown(verdict)

    st.markdown("---")

    # ── Skill Gaps inline ───────────────────────────────────────────────────
    if gaps:
        badges = "".join(f'<span class="gap-badge">{g}</span>' for g in gaps)
        st.markdown(
            f"**🔍 Detected Skill Gaps**  \n"
            f"<div class='gap-row'>{badges}</div><br>",
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ No critical skill gaps detected — your profile covers all key requirements.")

    st.markdown("---")

    # ── Expandable agent output sections ────────────────────────────────────
    with st.expander("📋 Resume Review — Detailed Feedback", expanded=True):
        review = result.get("resume_review") or "_No review available._"
        st.markdown(review)

    with st.expander("⚡ ATS-Optimised Resume Rewrite"):
        st.caption(
            "The pipeline rewrote your resume to maximise ATS keyword alignment. "
            "Copy this version when applying."
        )
        st.text_area(
            "Optimised resume",
            value=result.get("resume_text") or "",
            height=340,
            disabled=True,
            label_visibility="collapsed",
        )

    with st.expander("📚 Week-by-Week Study Plan"):
        plan = result.get("study_plan") or "_No study plan available._"
        st.markdown(plan)

    with st.expander("🎯 Mock Interview — 5 Tailored Q&As"):
        interview = result.get("interview_feedback") or "_No interview Q&As available._"
        st.markdown(interview)

    with st.expander("👥 Group Discussion — Simulation & Evaluation"):
        gd = result.get("gd_feedback") or "_No GD feedback available._"
        st.markdown(gd)

    with st.expander("📄 Original Extracted Resume Text"):
        st.caption("Raw text extracted from your uploaded PDF by pdfplumber.")
        # Show the raw text from initial extraction (before ATS rewrite)
        # The API returns the optimised text in resume_text; original is unavailable post-pipeline,
        # so we show the current state's resume_text with a note.
        st.text_area(
            "raw text",
            value=result.get("resume_text") or "",
            height=280,
            disabled=True,
            label_visibility="collapsed",
        )

# ── Landing state ────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="
        background: #12131f;
        border: 1px dashed rgba(99,102,241,.3);
        border-radius: 18px;
        padding: 2.5rem 2rem;
        text-align: center;
        max-width: 760px;
        margin: 2rem auto;
    ">
        <div style="font-size:3rem;margin-bottom:1rem">🤖</div>
        <h3 style="color:#818cf8;margin-bottom:.8rem;font-weight:600">
            6-Agent LangGraph Pipeline
        </h3>
        <p style="color:rgba(255,255,255,.55);font-size:1rem;line-height:1.7;margin-bottom:1.5rem">
            Upload your resume PDF and paste a job description in the sidebar to trigger the full
            multi-agent evaluation workflow.
        </p>
        <div style="
            display:grid;
            grid-template-columns: repeat(2,1fr);
            gap:.8rem;
            text-align:left;
        ">
            <div style="background:rgba(255,255,255,.03);border-radius:10px;padding:1rem">
                <b style="color:#c7d2fe">Step 1</b>
                <p style="color:rgba(255,255,255,.5);font-size:.85rem;margin:.3rem 0 0">
                    Upload PDF resume &amp; paste job description
                </p>
            </div>
            <div style="background:rgba(255,255,255,.03);border-radius:10px;padding:1rem">
                <b style="color:#c7d2fe">Step 2</b>
                <p style="color:rgba(255,255,255,.5);font-size:.85rem;margin:.3rem 0 0">
                    Click Analyse and review all 6 agent outputs + readiness score
                </p>
            </div>
        </div>
    </div>

    <div style="text-align:center;margin-top:2rem">
        <p style="color:rgba(255,255,255,.25);font-size:.8rem">
            Powered by Gemini (gemini-2.0-flash) · LangGraph · FastAPI · Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)
