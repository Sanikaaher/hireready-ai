from typing import TypedDict, List, Optional

class CandidateState(TypedDict):
    """
    State representing the candidate placement preparation process.
    """
    resume_text: str
    job_description: str
    ats_score: Optional[float]
    skill_gaps: List[str]
    study_plan: Optional[str]
    interview_feedback: Optional[str]
    gd_feedback: Optional[str]
    readiness_score: Optional[float]
    iteration_count: int
    resume_review: Optional[str]
