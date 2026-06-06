import os
import time
from typing import Dict, Any
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
        temperature=0.2
    )

def resume_reviewer_node(state: CandidateState) -> CandidateState:
    """
    Reviews candidate resume, returns structured feedback.
    Receives and returns CandidateState.
    """
    print("--- RUNNING RESUME REVIEWER AGENT NODE ---")
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")
    
    llm = get_llm()
    if not llm:
        # High-fidelity mock feedback fallback
        feedback = (
            "### Resume Review Feedback\n\n"
            "#### 1. Structure & Presentation\n"
            "- **Pros:** Clear section separation, standard reverse-chronological order.\n"
            "- **Areas for Improvement:** The resume contains too many text-dense paragraphs; use bullet points beginning with action verbs for better ATS readability.\n\n"
            "#### 2. Key Accomplishments & Impact\n"
            "- **Areas for Improvement:** Bullet points focus on responsibilities rather than quantitative achievements. (e.g., instead of 'Responsible for APIs', use 'Developed 15+ FastAPI endpoints, decreasing query latency by 25%').\n\n"
            "#### 3. Role Relevancy\n"
            "- **Analysis:** The technical background is strong in general software development but lacks specific highlights matching cloud and database requirements from the job description."
        )
    else:
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an expert Resume Reviewer and recruiter.\n"
                "Evaluate the candidate's resume based on the target job description.\n"
                "Provide professional, actionable, and structured feedback covering formatting, visual layout, content impact, and role alignment.\n\n"
                "Resume:\n{resume_text}\n\n"
                "Job Description:\n{job_description}"
            )
            chain = prompt | llm
            response = chain.invoke({"resume_text": resume_text, "job_description": job_description})
            time.sleep(4)
            feedback = response.content
        except Exception as e:
            feedback = f"Error in Resume Reviewer agent: {str(e)}"
            
    # Update state dict and return
    updated_state = dict(state)
    updated_state["resume_review"] = feedback
    return updated_state
