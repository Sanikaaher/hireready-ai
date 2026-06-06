import os
from typing import List
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
        temperature=0.4
    )

def _mock_study_plan(skill_gaps: List[str]) -> str:
    """High-fidelity mock study plan fallback."""
    if not skill_gaps:
        return (
            "### 📚 Study Plan\n\n"
            "✅ **No critical skill gaps detected.** Your profile aligns well with the job requirements.\n\n"
            "**Recommended next steps:**\n"
            "- Week 1–2: Deepen expertise in your strongest domain with advanced projects.\n"
            "- Week 3–4: Contribute to an open-source project relevant to the target role.\n"
            "- Week 5–6: Prepare system design case studies and behavioral interview stories.\n"
            "- Week 7–8: Do 2–3 mock interviews and get peer feedback."
        )

    lines = ["## 📅 Week-by-Week Study Plan\n"]
    resources = {
        "System Design": ("'Designing Data-Intensive Applications' by Martin Kleppmann", "system-design-primer on GitHub"),
        "Docker Containerization": ("Official Docker Docs (docs.docker.com)", "TechWorld with Nana – Docker Crash Course on YouTube"),
        "Cloud Deployment (AWS)": ("AWS Free Tier + official AWS Skill Builder", "A Cloud Guru – AWS Solutions Architect course"),
        "Database ORMs": ("SQLAlchemy official docs", "Full Stack Python – SQLAlchemy chapter"),
        "FastAPI": ("FastAPI official docs (fastapi.tiangolo.com)", "ArjanCodes – FastAPI tutorial series"),
        "LangChain / LangGraph": ("LangChain official docs (python.langchain.com)", "LangGraph tutorials on GitHub"),
    }

    for i, gap in enumerate(skill_gaps, 1):
        r1, r2 = resources.get(gap, (f"{gap} official documentation", f"YouTube tutorials on {gap}"))
        lines.append(f"### 🗓️ Week {i}: **{gap}**")
        lines.append(f"**Goal:** Achieve working-level proficiency in {gap}.")
        lines.append(f"\n**Daily Targets:**")
        lines.append(f"- Mon–Tue: Read core theory and official docs.")
        lines.append(f"- Wed–Thu: Build a mini hands-on project integrating {gap}.")
        lines.append(f"- Fri: Write a reflection blog post or README documenting what you built.")
        lines.append(f"- Sat–Sun: Code review, push to GitHub, and review next week's topic.\n")
        lines.append(f"**Primary Resources:**")
        lines.append(f"- 📖 {r1}")
        lines.append(f"- 🎥 {r2}\n")
        lines.append(f"**Expected Outcome:** A working demo / mini-project for {gap} on your GitHub profile.\n")
        lines.append("---\n")

    lines.append(
        "### 🏁 Final Week: Consolidation & Interview Prep\n"
        "- Revise all built projects.\n"
        "- Practice explaining each project in a 2-minute structured STAR format.\n"
        "- Do at least 2 timed system design mock sessions."
    )
    return "\n".join(lines)


def study_planner_node(state: CandidateState) -> CandidateState:
    """
    Generates a structured week-by-week study plan based on identified skill gaps.
    Receives and returns CandidateState.
    """
    print("--- RUNNING STUDY PLANNER AGENT NODE ---")
    skill_gaps: List[str] = state.get("skill_gaps", [])

    llm = get_llm()
    if not llm:
        plan = _mock_study_plan(skill_gaps)
    else:
        try:
            gaps_text = ", ".join(skill_gaps) if skill_gaps else "no critical gaps detected"
            prompt = ChatPromptTemplate.from_template(
                "You are an expert technical career coach and curriculum designer.\n"
                "A candidate is preparing for a software engineering role and has the following skill gaps:\n"
                "{gaps_text}\n\n"
                "Create a detailed, actionable **week-by-week study plan** to help them close these gaps.\n"
                "For each week:\n"
                "  1. Name the skill being covered.\n"
                "  2. List daily learning targets (Mon–Sun).\n"
                "  3. Recommend 2 specific, freely available resources (books, docs, YouTube channels, or courses).\n"
                "  4. Define a concrete mini-project or deliverable to complete by end of week.\n"
                "  5. Define the expected outcome / proficiency level.\n\n"
                "Format the plan in clean Markdown with week-by-week sections, emojis for visual clarity, "
                "and a final consolidation week at the end. Be specific — avoid generic advice."
            )
            chain = prompt | llm
            response = chain.invoke({"gaps_text": gaps_text})
            plan = response.content
        except Exception as e:
            plan = _mock_study_plan(skill_gaps)
            plan += f"\n\n> ⚠️ *Live generation error (using detailed fallback): {str(e)}*"

    updated_state = dict(state)
    updated_state["study_plan"] = plan
    return updated_state
