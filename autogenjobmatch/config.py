"""
Configuration settings for the job matching simulation.
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

# LLM Settings
GROQ_MODEL = "llama3-70b-8192"  # or another available model

# Simulation Parameters
NUM_CANDIDATES = 5  # Default number of candidates
NUM_COMPANIES = 3   # Default number of companies
INITIAL_ENERGY = 15  # Initial energy for candidates
INTERVIEW_ENERGY_COST = 1  # Energy cost per interview

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
DEFAULT_INTERVIEW_ROUNDS = 4  # Default number of interview rounds
MAX_INTERVIEW_ROUNDS = 6      # Maximum interview rounds
MIN_INTERVIEW_ROUNDS = 2      # Minimum interview rounds
INTERVIEW_STRATEGIES = ["Formal", "Casual"]
INTERVIEW_ATTITUDES = ["Humble", "Confident"]

# Probability parameters
LIE_PROBABILITY = 0.3  # Probability of candidates lying on their CV
VAGUE_JOB_DESCRIPTION_PROBABILITY = 0.4  # Probability of job descriptions being vague
EMOJI_PROBABILITY = 0.3  # Probability of job descriptions using emojis

# Salary ranges (in USD)
MIN_SALARY = 50000
MAX_SALARY = 150000

# Autogen execution configuration
CODE_EXECUTION_CONFIG = {
    "use_docker": False,  # Disable Docker requirement
    "work_dir": "workspace",  # Local directory for code execution
    "last_n_messages": 2,  # Number of messages to consider for code execution
}

# Output and logging
DEFAULT_OUTPUT_DIR = "results"
LOG_DIR = "logs"

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
Interview attitude: {attitude}

Ask relevant questions to assess the candidate's skills and fit for the position.
"""

# Default weights for job offer evaluation
EVALUATION_WEIGHTS = {
    "skill_match": 0.4,  # Weight for matching skills
    "sector_motivation": 0.3,  # Weight for candidate's motivation in the sector
    "salary_match": 0.3  # Weight for matching salary expectations
}

# Default weights for candidate evaluation
CANDIDATE_EVALUATION_WEIGHTS = {
    "skill_match": 0.7,  # Weight for matching skills
    "budget_fit": 0.3  # Weight for fitting within budget
}

# Function to get parameter with environment variable override
def get_param(name: str, default: Any) -> Any:
    """
    Get a parameter with potential environment variable override.
    
    Args:
        name: Parameter name
        default: Default value
        
    Returns:
        Parameter value (from environment if available, otherwise default)
    """
    env_var = f"JOBMATCH_{name.upper()}"
    env_value = os.getenv(env_var)
    
    if env_value is None:
        return default
        
    # Try to convert to the same type as default
    if isinstance(default, int):
        return int(env_value)
    elif isinstance(default, float):
        return float(env_value)
    elif isinstance(default, bool):
        return env_value.lower() in ("true", "yes", "1", "t", "y")
    elif isinstance(default, list):
        return env_value.split(",")
    else:
        return env_value