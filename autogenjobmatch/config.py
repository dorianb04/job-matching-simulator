"""
Configuration settings for the simulation.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

# LLM Settings
GROQ_MODEL = "llama3-70b-8192"

# Simulation Parameters
NUM_CANDIDATES = 5  # Start with a small number for testing
NUM_COMPANIES = 3
INITIAL_ENERGY = 15
INTERVIEW_ENERGY_COST = 1

# Business Sectors
SECTORS = [
    "Technology", "Finance", "Healthcare", "Education", 
    "Retail", "Manufacturing", "Entertainment"
]

# Data Science Skills
DATA_SCIENCE_SKILLS = [
    "Python", "R", "SQL", "Machine Learning", "Deep Learning", 
    "Data Visualization", "Statistics", "Big Data", "NLP"
]

# Interview Parameters
INTERVIEW_ROUNDS = 2
INTERVIEW_STRATEGIES = ["Formal", "Casual"]
INTERVIEW_ATTITUDES = ["Humble", "Confident"]

# Basic prompts
CANDIDATE_PROMPT = """
You are a job candidate with the following profile:
- Energy: {energy} points
- Motivation for different sectors: {motivation}
- Money importance: {money_importance}
- Data science skills: {skills}

Your goal is to find a job that matches your skills and preferences.
"""

COMPANY_PROMPT = """
You are a company in the {sector} sector named {name}.
Your budget for hiring is {budget}.
You're looking for candidates with these skills: {skills}.
"""

INTERVIEWER_PROMPT = """
You are an interviewer at {company_name}.
Your task is to interview candidates for a {position} position.
Interview style: {style}
"""

# Autogen execution configuration
CODE_EXECUTION_CONFIG = {
    "use_docker": False,  # Disable Docker requirement
    "work_dir": "workspace",  # Local directory for code execution
    "last_n_messages": 2,  # Number of messages to consider for code execution
}