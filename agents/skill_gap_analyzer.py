import os
import time
from typing import Dict, Any, List
from graph.state import CandidateState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define Pydantic schema for JSON parsing
class SkillGapSchema(BaseModel):
    ats_score: float = Field(description="ATS match score from 0 to 100 based on keyword and skill alignment")
    skill_gaps: List[str] = Field(description="List of technologies, soft skills, or experience areas required but missing/weak in the resume")

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your-"):
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api_key,
        temperature=0.1
    )

def skill_gap_analyzer_node(state: CandidateState) -> CandidateState:
    """
    Compares resume skills vs job requirements, lists gaps, and updates ATS score.
    Receives and returns CandidateState.
    """
    print("--- RUNNING SKILL GAP ANALYZER AGENT NODE ---")
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")
    
    llm = get_llm()
    if not llm:
        # High-fidelity mock analysis fallback
        score = 65.0
        gaps = ["System Design", "Cloud Deployment (AWS)", "Docker Containerization", "Database ORMs"]
    else:
        try:
            parser = JsonOutputParser(pydantic_object=SkillGapSchema)
            prompt = ChatPromptTemplate.from_template(
                "You are an AI Technical Evaluator.\n"
                "Compare the candidate's resume text against the target job description requirements.\n"
                "Extract missing competencies/technologies and calculate an overall compatibility score (0 to 100).\n\n"
                "Resume:\n{resume_text}\n\n"
                "Job Description:\n{job_description}\n\n"
                "Format rules:\n{format_instructions}"
            )
            
            chain = prompt | llm | parser
            result = chain.invoke({
                "resume_text": resume_text,
                "job_description": job_description,
                "format_instructions": parser.get_format_instructions()
            })
            time.sleep(4)
            score = result.get("ats_score", 50.0)
            gaps = result.get("skill_gaps", [])
        except Exception as e:
            score = 50.0
            gaps = [f"Analyzer error: {str(e)}"]
            
    updated_state = dict(state)
    updated_state["ats_score"] = score
    updated_state["skill_gaps"] = gaps
    return updated_state
