from langgraph.graph import StateGraph, START, END
from graph.state import CandidateState
from agents import (
    resume_reviewer_node,
    ats_optimizer_node,
    skill_gap_analyzer_node,
    study_planner_node,
    interviewer_node,
    gd_moderator_node,
)


# ─────────────────────────────────────────────
# Critic Node
# ─────────────────────────────────────────────
def critic_node(state: CandidateState) -> CandidateState:
    """
    Evaluates overall candidate readiness after all 6 agents have run.
    Calculates readiness_score from ats_score and skill_gap count,
    then increments iteration_count. The downstream conditional router
    uses these values to decide whether to loop or terminate.
    """
    print("--- RUNNING CRITIC NODE ---")
    ats_score = state.get("ats_score", 0.0) or 0.0
    skill_gaps = state.get("skill_gaps", [])
    iteration_count = state.get("iteration_count", 0)

    # Each unresolved skill gap applies a 5-point penalty to readiness
    gap_penalty = len(skill_gaps) * 5
    readiness_score = max(0.0, min(100.0, ats_score - gap_penalty))

    updated_state = dict(state)
    updated_state["readiness_score"] = round(readiness_score, 1)
    updated_state["iteration_count"] = iteration_count + 1
    return updated_state


# ─────────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────────
def should_loop(state: CandidateState) -> str:
    """
    Routes back to 'resume_reviewer' for another improvement pass if:
      - readiness_score is below 70, AND
      - iteration_count is below 3 (prevents infinite loops).
    Otherwise routes to END.
    """
    readiness = state.get("readiness_score", 0.0) or 0.0
    iterations = state.get("iteration_count", 0)

    if readiness < 70.0 and iterations < 3:
        print(f"  ↺  Critic: score={readiness:.1f}, iter={iterations} → looping back to resume_reviewer")
        return "resume_reviewer"

    print(f"  ✓  Critic: score={readiness:.1f}, iter={iterations} → pipeline complete")
    return END


# ─────────────────────────────────────────────
# Build the StateGraph
# ─────────────────────────────────────────────
_builder = StateGraph(CandidateState)

# Register all 6 agent nodes + critic
_builder.add_node("resume_reviewer",   resume_reviewer_node)
_builder.add_node("ats_optimizer",     ats_optimizer_node)
_builder.add_node("skill_gap_analyzer", skill_gap_analyzer_node)
_builder.add_node("study_planner",     study_planner_node)
_builder.add_node("interviewer",       interviewer_node)
_builder.add_node("gd_moderator",      gd_moderator_node)
_builder.add_node("critic",            critic_node)

# Linear chain: START → 6 agents → critic
_builder.add_edge(START,               "resume_reviewer")
_builder.add_edge("resume_reviewer",   "ats_optimizer")
_builder.add_edge("ats_optimizer",     "skill_gap_analyzer")
_builder.add_edge("skill_gap_analyzer","study_planner")
_builder.add_edge("study_planner",     "interviewer")
_builder.add_edge("interviewer",       "gd_moderator")
_builder.add_edge("gd_moderator",      "critic")

# Conditional edge: critic → loop back OR END
_builder.add_conditional_edges(
    "critic",
    should_loop,
    {
        "resume_reviewer": "resume_reviewer",
        END: END,
    },
)

# Compile into a runnable LangGraph app
app = _builder.compile()
