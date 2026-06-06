import os
from agents.utils import invoke_with_retry
import pdfplumber
from typing import Dict, Any, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Initialize LLM helper
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your-"):
        # Returns None to signal fallback to mock data
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api_key,
        temperature=0.2
    )

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text from a PDF file using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        text = f"Error extracting PDF: {str(e)}"
    return text.strip()

# Pydantic schemas for structured LLM outputs
class ATSAnalysis(BaseModel):
    ats_score: float = Field(description="ATS compatibility score from 0 to 100")
    skill_gaps: List[str] = Field(description="List of key technical or soft skills missing or weak in the resume compared to the job description")

class PlacementAgents:
    @staticmethod
    def analyze_resume(resume_text: str, job_description: str) -> Tuple[float, List[str]]:
        """
        Analyzes the resume against the job description to calculate an ATS score and identify skill gaps.
        """
        llm = get_llm()
        if not llm:
            # High-quality fallback mock data
            return 72.5, ["System Design", "AWS Deployment", "Advanced Algorithms", "FastAPI middleware"]

        try:
            parser = JsonOutputParser(pydantic_object=ATSAnalysis)
            prompt = ChatPromptTemplate.from_template(
                "You are an expert ATS (Applicant Tracking System) parser and technical recruiter.\n"
                "Analyze the candidate's resume text against the job description.\n"
                "Provide a match score (0-100) and list the critical skill gaps.\n\n"
                "Resume:\n{resume_text}\n\n"
                "Job Description:\n{job_description}\n\n"
                "Format rules:\n{format_instructions}"
            )
            
            chain = prompt | llm | parser
            result = invoke_with_retry(chain, {
                "resume_text": resume_text,
                "job_description": job_description,
                "format_instructions": parser.get_format_instructions()
            })
            return result.get("ats_score", 50.0), result.get("skill_gaps", [])
        except Exception as e:
            print(f"Error invoking ATS agent: {e}")
            return 50.0, ["Error parsing details. Please check API Key."]

    @staticmethod
    def generate_study_plan(skill_gaps: List[str]) -> str:
        """
        Generates a targeted weekly preparation plan to cover the identified skill gaps.
        """
        llm = get_llm()
        if not llm:
            # High-quality fallback mock data
            plan_lines = ["### Weekly Study Plan\n"]
            for i, gap in enumerate(skill_gaps, 1):
                plan_lines.append(f"**Week {i}: Master {gap}**")
                plan_lines.append(f"- *Resources:* Free online documentation & practical hands-on mini-projects.")
                plan_lines.append(f"- *Goal:* Build a demo app/script integrating {gap} and post to GitHub.\n")
            return "\n".join(plan_lines)

        try:
            prompt = ChatPromptTemplate.from_template(
                "You are a technical mentor. Generate a structured, step-by-step weekly preparation plan "
                "to help a student master the following skill gaps: {skill_gaps}.\n"
                "Provide concrete actionable resources, daily targets, and expected outcomes in Markdown format."
            )
            chain = prompt | llm
            response = invoke_with_retry(chain, {"skill_gaps": ", ".join(skill_gaps)})
            return response.content
        except Exception as e:
            return f"Error generating study plan: {e}"

    @staticmethod
    def generate_interview_feedback(resume_text: str, job_description: str) -> str:
        """
        Simulates an interview assessment, suggesting key behavioral/technical questions
        and model answers customized to the candidate's background.
        """
        llm = get_llm()
        if not llm:
            return (
                "### Mock Interview Insights\n"
                "1. **Tell me about a time you worked under pressure.**\n"
                "   - *Expected response direction:* Highlight structured problem solving, delegation, and clear communication.\n"
                "2. **How would you structure a Python API using FastAPI for high concurrency?**\n"
                "   - *Expected response direction:* Discuss async/await endpoints, connection pooling, background tasks, and deployment with Uvicorn/Gunicorn."
            )

        try:
            prompt = ChatPromptTemplate.from_template(
                "You are a Senior Technical Interviewer.\n"
                "Based on the candidate's resume and target job description, generate 3 highly targeted interview questions.\n"
                "For each question, provide a brief 'Ideal Candidate Answer Guide'.\n\n"
                "Resume:\n{resume_text}\n\n"
                "Job Description:\n{job_description}"
            )
            chain = prompt | llm
            response = invoke_with_retry(chain, {"resume_text": resume_text, "job_description": job_description})
            return response.content
        except Exception as e:
            return f"Error simulating interview preparation: {e}"

    @staticmethod
    def generate_gd_feedback(job_description: str) -> str:
        """
        Provides guidance on Group Discussion topics and mock feedback strategies.
        """
        llm = get_llm()
        if not llm:
            return (
                "### Group Discussion Tips for Recruiter Panel\n"
                "- **Suggested Topic:** 'Is AI replacing junior software engineers, or scaling them up?'\n"
                "- **Key points to mention:** Code copilots increasing velocity, the rising importance of system architecture, security and compliance.\n"
                "- **Evaluation criteria:** Ability to listen, structure arguments logically, and collaborate rather than dominate."
            )

        try:
            prompt = ChatPromptTemplate.from_template(
                "You are a Placement Coordinator.\n"
                "Based on the job description, suggest a relevant Group Discussion (GD) topic.\n"
                "Provide 3 arguments for, 3 arguments against, and communication tips to succeed in this GD.\n\n"
                "Job Description:\n{job_description}"
            )
            chain = prompt | llm
            response = invoke_with_retry(chain, {"job_description": job_description})
            return response.content
        except Exception as e:
            return f"Error generating GD tips: {e}"
