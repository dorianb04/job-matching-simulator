"""
Functions to generate data for the simulation.
"""
import random
from typing import Dict, Any, List, Optional
import autogen
from .agents import create_user_proxy
from .config import (
    SECTORS, DATA_SCIENCE_SKILLS, 
    INITIAL_ENERGY, INTERVIEW_STRATEGIES,
    CANDIDATE_PROMPT, COMPANY_PROMPT
)
from .monitoring import AgentOpsTracker

def generate_candidate_profile(
    candidate_id: int,
    llm_config: Dict[str, Any],
    tracker: Optional[AgentOpsTracker] = None
) -> Dict[str, Any]:
    """
    Generate a candidate profile.
    
    Args:
        candidate_id: Unique identifier
        llm_config: LLM configuration
        tracker: AgentOps tracker
        
    Returns:
        Dictionary with candidate profile
    """
    # Track operation start
    if tracker:
        tracker.track_event("profile_generation_started", {
            "candidate_id": candidate_id,
            "type": "candidate"
        })
    
    # Create basic profile with random values
    motivation = {sector: random.randint(1, 10) for sector in SECTORS}
    skills = {skill: random.randint(0, 10) for skill in DATA_SCIENCE_SKILLS}
    money_importance = round(random.uniform(0.1, 1.0), 2)
    
    profile = {
        "id": candidate_id,
        "energy": INITIAL_ENERGY,
        "motivation": motivation,
        "money_importance": money_importance,
        "skills": skills
    }
    
    # Track operation completion
    if tracker:
        tracker.track_event("profile_generation_completed", {
            "candidate_id": candidate_id,
            "profile": profile
        })
    
    return profile

def generate_company_profile(
    company_id: int,
    llm_config: Dict[str, Any],
    tracker: Optional[AgentOpsTracker] = None
) -> Dict[str, Any]:
    """
    Generate a company profile.
    
    Args:
        company_id: Unique identifier
        llm_config: LLM configuration
        tracker: AgentOps tracker
        
    Returns:
        Dictionary with company profile
    """
    # Track operation start
    if tracker:
        tracker.track_event("profile_generation_started", {
            "company_id": company_id,
            "type": "company"
        })
    
    # Select random sector and create company name
    sector = random.choice(SECTORS)
    company_generator = autogen.AssistantAgent(
        name="company_name_generator",
        llm_config=llm_config,
        system_message="You generate realistic company names for given sectors."
    )
    
    user_proxy = create_user_proxy(name="user_proxy")
    
    # Generate company name
    user_proxy.initiate_chat(
        company_generator, 
        message=f"Generate a realistic company name for a {sector} company. Response format: just the name only."
    )
    company_name = company_generator.last_message()["content"].strip()
    
    # Select random required skills (3-5 skills)
    num_skills = random.randint(3, 5)
    required_skills = random.sample(DATA_SCIENCE_SKILLS, num_skills)
    
    # Generate random budget
    budget = random.randint(50000, 150000)
    
    profile = {
        "id": company_id,
        "name": company_name,
        "sector": sector,
        "budget": budget,
        "required_skills": required_skills
    }
    
    # Track operation completion
    if tracker:
        tracker.track_event("profile_generation_completed", {
            "company_id": company_id,
            "profile": profile
        })
    
    return profile

def generate_job_offer(
    company_profile: Dict[str, Any],
    job_id: int,
    llm_config: Dict[str, Any],
    tracker: Optional[AgentOpsTracker] = None
) -> Dict[str, Any]:
    """
    Generate a job offer for a company.
    
    Args:
        company_profile: Company profile
        job_id: Unique job identifier
        llm_config: LLM configuration
        tracker: AgentOps tracker
        
    Returns:
        Dictionary with job offer details
    """
    # Track operation start
    if tracker:
        tracker.track_event("job_offer_generation_started", {
            "company_id": company_profile["id"],
            "job_id": job_id
        })
    
    # Create job offer generator agent
    job_generator = autogen.AssistantAgent(
        name="job_offer_generator",
        llm_config=llm_config,
        system_message="You create realistic job offers based on company information."
    )
    
    user_proxy = create_user_proxy(name="user_proxy")
    
    # Generate job title
    user_proxy.initiate_chat(
        job_generator, 
        message=f"""
        Create a job title for a position at {company_profile['name']} (a {company_profile['sector']} company) 
        that requires these skills: {', '.join(company_profile['required_skills'])}.
        Response format: just the job title only.
        """
    )
    job_title = job_generator.last_message()["content"].strip()
    
    # Generate salary (within company budget range)
    budget = company_profile["budget"]
    min_salary = int(budget * 0.8)
    max_salary = int(budget * 1.1)
    
    # Determine if job description should be vague
    is_vague = random.choice([True, False])
    uses_emojis = random.choice([True, False])
    
    job_offer = {
        "id": job_id,
        "company_id": company_profile["id"],
        "company_name": company_profile["name"],
        "sector": company_profile["sector"],
        "title": job_title,
        "required_skills": company_profile["required_skills"],
        "salary_range": (min_salary, max_salary),
        "is_vague": is_vague,
        "uses_emojis": uses_emojis
    }
    
    # Track operation completion
    if tracker:
        tracker.track_event("job_offer_generation_completed", {
            "job_id": job_id,
            "job_offer": job_offer
        })
    
    return job_offer