import os
from typing import Dict, Any
from graph.state import CandidateState
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("your-"):
        return None
    try:
        return ChatAnthropic(
            model="claude-sonnet-4-5-20251022",
            anthropic_api_key=api_key,
            temperature=0.3
        )
    except Exception:
        return ChatAnthropic(
            model="claude-sonnet-4-5-20251022",
            anthropic_api_key=api_key,
            temperature=0.3
        )

def ats_optimizer_node(state: CandidateState) -> CandidateState:
    """
    Rewrites resume for the given job description to optimize ATS parsing.
    Receives and returns CandidateState.
    """
    print("--- RUNNING ATS OPTIMIZER AGENT NODE ---")
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")
    
    llm = get_llm()
    if not llm:
        # High-fidelity mock optimizer fallback
        optimized = (
            f"=== OPTIMIZED RESUME FOR ATS CONTEXT ===\n\n"
            f"{resume_text}\n\n"
            f"--- ATS OPTIMIZED KEYWORD PROFILE INTEGRATED ---\n"
            f"Technical Keywords: FastAPI, REST APIs, PostgreSQL, AWS Deployment, System Design, Python Software Engineering\n"
            f"Action Keywords: Developed, Configured, Designed, Optimized, Integrated"
        )
    else:
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an expert ATS optimization consultant.\n"
                "Rewrite and polish the candidate's resume to optimize it for the target job description.\n"
                "Incorporate relevant skills, industry keywords, and vocabulary mentioned in the job description.\n"
                "IMPORTANT: Retain all actual historical details, projects, and work experiences from the original resume. Do not invent fake achievements or job titles.\n\n"
                "Original Resume:\n{resume_text}\n\n"
                "Target Job Description:\n{job_description}"
            )
            chain = prompt | llm
            response = chain.invoke({"resume_text": resume_text, "job_description": job_description})
            optimized = response.content
        except Exception as e:
            optimized = f"Error in ATS Optimizer agent: {str(e)}"
            
    updated_state = dict(state)
    updated_state["resume_text"] = optimized
    return updated_state
