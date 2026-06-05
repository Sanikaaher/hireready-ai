from langgraph.graph import StateGraph, START, END
from graph.state import CandidateState
from graph.nodes import (
    study_plan_node,
    interview_feedback_node,
    gd_feedback_node,
    readiness_node
)
from agents import (
    resume_reviewer_node,
    ats_optimizer_node,
    skill_gap_analyzer_node
)

# Initialize workflow with our custom CandidateState TypedDict
workflow = StateGraph(CandidateState)

# Register nodes
workflow.add_node("resume_reviewer", resume_reviewer_node)
workflow.add_node("ats_optimizer", ats_optimizer_node)
workflow.add_node("skill_gap_analyzer", skill_gap_analyzer_node)
workflow.add_node("study_plan", study_plan_node)
workflow.add_node("interview_feedback", interview_feedback_node)
workflow.add_node("gd_feedback", gd_feedback_node)
workflow.add_node("readiness", readiness_node)

# Define edges
workflow.add_edge(START, "resume_reviewer")
workflow.add_edge("resume_reviewer", "ats_optimizer")
workflow.add_edge("ats_optimizer", "skill_gap_analyzer")
workflow.add_edge("skill_gap_analyzer", "study_plan")
workflow.add_edge("study_plan", "interview_feedback")
workflow.add_edge("interview_feedback", "gd_feedback")
workflow.add_edge("gd_feedback", "readiness")
workflow.add_edge("readiness", END)

# Compile workflow
app = workflow.compile()
