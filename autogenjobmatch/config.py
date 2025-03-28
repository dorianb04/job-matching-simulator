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
GROQ_MODEL = "llama-3.1-8b-instant"  # or another available model

# Simulation Parameters
NUM_CANDIDATES = 5  # Default number of candidates
NUM_COMPANIES = 3   # Default number of companies
NUM_SIMULATION_ROUNDS = 3  # Number of rounds to run the simulation
INITIAL_ENERGY = 15  # Initial energy for candidates
INTERVIEW_ENERGY_COST = 1  # Energy cost per interview

# Strategic Parameters
STRATEGIC_LEARNING_RATE = 0.1  # How quickly agents adapt
MAX_APPLICATIONS_PER_CANDIDATE = 15  # Maximum applications per candidate
MAX_INTERVIEWS_PER_COMPANY = 4  # Maximum interviews per company
MAX_OFFERS_PER_COMPANY = 2  # Maximum offers per company

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

# Universities
UNIVERSITIES = [
    "École Polytechnique", 
    "Université Paris-Saclay", 
    "Sorbonne Université", 
    "Sciences Po", 
    "CentraleSupélec", 
    "École Normale Supérieure Paris", 
    "HEC Paris", 
    "ESSEC Business School", 
    "ENSAE Paris", 
    "Université Paris-Dauphine", 
    "Université Grenoble Alpes", 
    "Université de Strasbourg",
    "EDHEC Business School",
    "ESCP Business School", 
    "Institut Polytechnique de Paris",
    "École des Ponts ParisTech",
    "INSA Lyon",
    "Toulouse Business School",
    "IMT Atlantique",
    "Université Claude Bernard Lyon 1",
    "IUT Lyon 2",
    "HEC Montréal",
    "HEC Lausanne",
    "Bocconi University",
    "Polytech Nice"
]

# European Countries
EUROPEAN_COUNTRIES = [
    "France"
]

# Interview Parameters
DEFAULT_INTERVIEW_ROUNDS = 4  # Default number of interview rounds
MAX_INTERVIEW_ROUNDS = 6      # Maximum interview rounds
MIN_INTERVIEW_ROUNDS = 2      # Minimum interview rounds
INTERVIEW_STRATEGIES = ["Technical", "Behavioral"]
INTERVIEW_ATTITUDES = ["Formal", "Casual"]

# Probability parameters
LIE_PROBABILITY = 0.3  # Probability of candidates lying on their CV
VAGUE_JOB_DESCRIPTION_PROBABILITY = 0.4  # Probability of job descriptions being vague
EMOJI_PROBABILITY = 0.3  # Probability of job descriptions using emojis

# Salary ranges (in euros)
MIN_SALARY = 30000
MAX_SALARY = 120000

# Autogen execution configuration
CODE_EXECUTION_CONFIG = {
    "use_docker": False,  # Disable Docker requirement
    "work_dir": "workspace",  # Local directory for code execution
    "last_n_messages": 2,  # Number of messages to consider for code execution
}

# Output and logging
DEFAULT_OUTPUT_DIR = "results"
LOG_DIR = "logs"

# Enhanced evaluation weights
EVALUATION_WEIGHTS = {
    "skill_match": 0.4,  # Weight for matching skills
    "sector_motivation": 0.3,  # Weight for candidate's motivation in the sector
    "salary_match": 0.2,  # Weight for matching salary expectations
    "strategic_bonus": 0.1  # Weight for strategic considerations
}

# Enhanced candidate evaluation weights
CANDIDATE_EVALUATION_WEIGHTS = {
    "skill_match": 0.7,  # Weight for matching skills
    "budget_fit": 0.3  # Weight for fitting within budget
}

# Reward weights for candidates
CANDIDATE_REWARD_WEIGHTS = {
    "salary": 0.5,  # Weight for salary relative to market
    "sector_alignment": 0.3,  # Weight for matching preferred sectors
    "skill_utilization": 0.2  # Weight for using existing skills
}

# Reward weights for companies
COMPANY_REWARD_WEIGHTS = {
    "skill_match": 0.6,  # Weight for matching required skills
    "budget_efficiency": 0.3,  # Weight for salary vs. budget
    "retention_probability": 0.1  # Weight for estimated retention
}

# Enhanced prompts
CANDIDATE_PROMPT = """
You are a job candidate with the following profile:
- Energy: {energy} points
- Motivation for different sectors: {motivation}
- Money importance: {money_importance}
- Data science skills: {skills}
- Risk tolerance: {risk_tolerance} (higher means more willing to take risks)
- Truth level: {truth_level} (higher means more honest in applications)
- Interview style: {interview_style} (Formal or Casual)
- Interview attitude: {interview_attitude} (Humble or Confident)
- Lyingness: {lyingness} (0.0 = completely honest, 0.5 = moderately exaggerates, 1.0 = fabricates extensively)
- Skills to improve: {skills_to_improve}

Your goal is to find a job that matches your skills and preferences while maximizing your career potential.

In interviews, your communication style should be {interview_style} and your attitude should be {interview_attitude}.

IMPORTANT ABOUT LYINGNESS:
- With lyingness = 0.0: Be completely honest about your skills and experience
- With lyingness = 0.5: Moderately exaggerate your achievements and skills
- With lyingness = 1.0: Fabricate skills and experience substantially

When discussing skills where you've exaggerated or lied:
- Show appropriate hesitation when asked detailed questions
- If your true skill level in something is low but you claimed high expertise, you'll struggle to answer technical questions
- The more you've exaggerated a skill, the less confident your answers should be about it
- If directly questioned about inconsistencies, either deflect or admit to "overstating"

Use your judgment to highlight your strengths and address any weaknesses strategically.
"""

COMPANY_PROMPT = """
You are a company in the {sector} sector named {name}.
Your budget for hiring is {budget}.
You're looking for candidates with these skills: {skills}.
Your hiring standards are {hiring_standards} (higher means more selective).
Your budget flexibility is {budget_flexibility} (how much above budget you can go).

Your goal is to find the best candidates who match your skill requirements while staying within your budget constraints.

In interviews, assess candidates thoroughly for both technical skills and cultural fit.
"""

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