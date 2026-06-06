import os
from agents.utils import invoke_with_retry
from graph.state import CandidateState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your-"):
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api_key,
        temperature=0.6
    )

_MOCK_GD = """## 👥 Group Discussion Simulation

---

### 🗣️ Topic: *"Is AI replacing junior software engineers, or empowering them to do more?"*

*Relevance to Role: This topic directly connects to the evolving landscape of backend engineering,
AI tooling adoption, and the candidate's ability to articulate nuanced technical positions.*

---

### 📋 Simulated Discussion Transcript

**Moderator:** "Let's begin. The motion is: AI is net-positive for junior software engineers in 2025. Please share your perspectives."

---

**Participant A — Priya (Advocate):**
"I believe AI is a force multiplier. Tools like GitHub Copilot cut boilerplate time by 40%, letting
juniors focus on architecture and problem-solving rather than syntax. I've seen teammates ship
production-ready APIs in half the time. AI raises the floor, not the ceiling."

**Participant B — Raj (Challenger):**
"That's optimistic, but there's a risk. If juniors rely on AI for every line of code, they miss
foundational debugging skills. When the AI hallucinates a SQL query and you can't read the execution
plan, that's a production outage waiting to happen. Tooling without understanding is dangerous."

**Participant C — Maya (Synthesiser):**
"Both of you have a point. The key variable is *intentionality*. Engineers who use AI as a learning
companion — asking it to explain generated code, not just copy it — grow faster. Those who use it as
a crutch stagnate. Companies need mentorship structures that reward curiosity, not just output velocity."

**Moderator:** "Excellent. Any closing statements?"

**Participant A:** "Upskill with AI, don't outsource thinking to it."
**Participant B:** "Gate AI access behind fundamentals training."
**Participant C:** "Build a culture of 'explain-the-AI' code reviews."

---

### 📊 Moderator's Evaluation

| Criteria | Assessment |
|---|---|
| **Clarity of Argument** | Strong — each participant held a coherent position |
| **Listening & Building on Others** | Maya effectively synthesised the prior two views |
| **Use of Evidence** | Priya cited a specific metric (40%); Raj used a concrete failure scenario |
| **Collaboration vs. Domination** | Balanced — no participant talked over another |
| **Conclusion Quality** | All three gave concise, memorable closing lines |

---

### 💬 Tips to Excel in Your GD

- **Open strongly** — be the first or second to speak with a clear stance.
- **Use data or examples** — generic opinions lose to specific, verifiable claims.
- **Bridge, don't interrupt** — say "Building on what Raj said…" to show active listening.
- **Summarise at the end** — moderators love candidates who synthesise the group's ideas.
- **Stay calm under pushback** — disagreement handled gracefully signals leadership potential.
"""

def gd_moderator_node(state: CandidateState) -> CandidateState:
    """
    Simulates a group discussion relevant to the job description, complete with
    a multi-participant transcript and moderator's evaluation rubric.
    Receives and returns CandidateState.
    """
    print("--- RUNNING GD MODERATOR AGENT NODE ---")
    job_description = state.get("job_description", "")

    llm = get_llm()
    if not llm:
        feedback = _MOCK_GD
    else:
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an expert Group Discussion (GD) Moderator and placement coach.\n"
                "Based on the job description below, perform the following:\n\n"
                "1. **Choose a relevant GD topic** — one that connects to the industry/role in the JD "
                "and would realistically appear in a campus placement or lateral hiring GD round.\n\n"
                "2. **Simulate a short GD transcript** (5–7 exchanges) between 3 named participants. "
                "Each participant should hold a distinct stance: one advocate, one challenger, one synthesiser. "
                "The dialogue must feel natural and intellectually rigorous.\n\n"
                "3. **Write a Moderator's Evaluation** after the transcript using a Markdown table that scores "
                "the group discussion on: Clarity of Argument, Listening & Building on Others, "
                "Use of Evidence/Examples, Collaboration vs Domination, and Conclusion Quality.\n\n"
                "4. **List 5 actionable tips** for the candidate to stand out in this specific GD topic.\n\n"
                "Job Description:\n{job_description}\n\n"
                "Format the entire output in clean, structured Markdown. "
                "Use section headers, bold text, and tables for readability."
            )
            chain = prompt | llm
            response = invoke_with_retry(chain, {"job_description": job_description})
            feedback = response.content
        except Exception as e:
            feedback = _MOCK_GD
            feedback += f"\n\n> ⚠️ *Live generation error (using detailed fallback): {str(e)}*"

    updated_state = dict(state)
    updated_state["gd_feedback"] = feedback
    return updated_state
