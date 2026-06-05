from graph.state import CandidateState
from agents.placement_agents import PlacementAgents

def ats_score_node(state: CandidateState) -> dict:
    """
    Evaluates the resume against the job description to output the ATS score and identify skill gaps.
    """
    print("--- RUNNING ATS ANALYSIS NODE ---")
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")
    
    score, gaps = PlacementAgents.analyze_resume(resume_text, job_description)
    
    return {
        "ats_score": score,
        "skill_gaps": gaps
    }

def study_plan_node(state: CandidateState) -> dict:
    """
    Generates a weekly study plan using the identified skill gaps.
    """
    print("--- RUNNING STUDY PLAN NODE ---")
    skill_gaps = state.get("skill_gaps", [])
    
    if not skill_gaps:
        plan = "No significant skill gaps found. Continue regular preparation!"
    else:
        plan = PlacementAgents.generate_study_plan(skill_gaps)
        
    return {
        "study_plan": plan
    }

def interview_feedback_node(state: CandidateState) -> dict:
    """
    Generates interview questions and feedback mock details.
    """
    print("--- RUNNING INTERVIEW PREPARATION NODE ---")
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")
    
    feedback = PlacementAgents.generate_interview_feedback(resume_text, job_description)
    
    return {
        "interview_feedback": feedback
    }

def gd_feedback_node(state: CandidateState) -> dict:
    """
    Generates group discussion pointers and guides.
    """
    print("--- RUNNING GD FEEDBACK NODE ---")
    job_description = state.get("job_description", "")
    
    feedback = PlacementAgents.generate_gd_feedback(job_description)
    
    return {
        "gd_feedback": feedback
    }

def readiness_node(state: CandidateState) -> dict:
    """
    Calculates final readiness score and updates iteration count.
    """
    print("--- RUNNING READINESS ASSESSMENT NODE ---")
    ats_score = state.get("ats_score", 0.0) or 0.0
    skill_gaps = state.get("skill_gaps", [])
    current_iterations = state.get("iteration_count", 0)
    
    # Simple logic for readiness scoring
    gap_penalty = len(skill_gaps) * 5
    readiness_score = max(0.0, min(100.0, ats_score - gap_penalty))
    
    return {
        "readiness_score": readiness_score,
        "iteration_count": current_iterations + 1
    }
