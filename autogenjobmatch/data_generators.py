"""
Data generators for the job matching simulation.
"""
import random
import time
from typing import Dict, Any, List, Optional
import autogen

from .config import (
    SECTORS,
    DATA_SCIENCE_SKILLS,
    INITIAL_ENERGY,
    MIN_SALARY,
    MAX_SALARY,
    VAGUE_JOB_DESCRIPTION_PROBABILITY,
    EMOJI_PROBABILITY,
    LIE_PROBABILITY,
    UNIVERSITIES
)
from .agents import create_user_proxy

# Try to import AgentOps for tracking (optional)
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False

@agentops.record_action('generate_candidate_profile') if AGENTOPS_AVAILABLE else lambda func: func
def generate_candidate_profile(
    candidate_id: int,
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a candidate profile with strategic parameters.
    
    Args:
        candidate_id: Unique identifier
        llm_config: LLM configuration
        
    Returns:
        Dictionary with candidate profile
    """
    # Create personality traits
    motivation = {sector: random.randint(1, 10) for sector in SECTORS}
    
    # Generate skills
    skills = {}
    num_strong_skills = random.randint(2, 4)
    strong_skills = random.sample(DATA_SCIENCE_SKILLS, num_strong_skills)
    
    for skill in DATA_SCIENCE_SKILLS:
        if skill in strong_skills:
            # Strong skills - levels 6-10
            skills[skill] = random.randint(5, 10)
        else:
            # Other skills - levels 0-5
            skills[skill] = random.randint(0, 5)
    
    # Assign random French university
    university = random.choice(UNIVERSITIES)
    
    profile = {
        "id": candidate_id,
        "energy": INITIAL_ENERGY,
        "motivation": motivation,
        "money_importance": round(random.uniform(0.1, 1.0), 2),
        "skills": skills,
        "initial_skills": skills.copy(),  # Store initial skills for comparison
        "risk_tolerance": round(random.uniform(0.2, 0.9), 2),
        "work_life_balance": round(random.uniform(0.3, 1.0), 2),
        "education": {
            "university": university,
            "degree": random.choice(["Bachelor's", "Master's", "PhD"]),
            "field": random.choice(["Computer Science", "Data Science", "Statistics", "Mathematics", "Engineering"])
        },
        "career_goals": random.sample([
            "Technical expertise", "Management", "Innovation", "Job security",
            "Remote work", "International experience", "Work-life balance"
        ], 2),
        "total_applications": 0,
        "interviews_attended": 0,
        "offers_received": 0
    }
    
    return profile

@agentops.record_action('generate_company_profile') if AGENTOPS_AVAILABLE else lambda func: func
def generate_company_profile(
    company_id: int,
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a company profile with more cultural elements.
    
    Args:
        company_id: Unique identifier
        llm_config: LLM configuration
        
    Returns:
        Dictionary with company profile
    """
    start_time = time.time()
    
    # Track operation with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"generating_company:{company_id}"])
        except:
            pass
    
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
    
    # Generate random budget - now with more variation by sector
    if sector in ["Finance", "Technology", "Healthcare"]:
        # Higher paying sectors
        budget = random.randint(int(MIN_SALARY * 1.2), MAX_SALARY)
    else:
        # More moderate paying sectors
        budget = random.randint(MIN_SALARY, int(MAX_SALARY * 0.8))
    
    # Company culture elements
    culture_types = ["Innovative", "Traditional", "Collaborative", "Competitive", "Flexible"]
    company_size = random.choice(["Startup", "Small", "Medium", "Large", "Enterprise"])
    work_model = random.choice(["Remote", "Hybrid", "In-office"])
    
    # Generate company profile
    profile = {
        "id": company_id,
        "name": company_name,
        "sector": sector,
        "budget": budget,
        "required_skills": required_skills,
        "culture": random.choice(culture_types),
        "size": company_size,
        "work_model": work_model,
        "reputation": random.uniform(3.0, 5.0),  # Company rating out of 5
        "hiring_standards": random.uniform(0.5, 0.9),  # How selective
        "total_positions": random.randint(1, 3),  # How many positions available
        "applicants_hired": 0
    }
    
    # Track completion with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            generation_time = time.time() - start_time
            agentops.add_tags([
                f"company_generated:{company_id}",
                f"company_sector:{sector}",
                f"generation_time:{generation_time:.2f}s"
            ])
        except:
            pass
    
    return profile

@agentops.record_action('generate_job_description') if AGENTOPS_AVAILABLE else lambda func: func
def generate_job_description(
    job_listing: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> str:
    """
    Generate a detailed job description for a job listing.
    
    Args:
        job_listing: Job listing data
        llm_config: LLM configuration
        
    Returns:
        Job description text
    """
    start_time = time.time()
    job_id = job_listing["id"]
    company_id = job_listing["company_id"]
    
    # Track operation with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"generating_job_description:{job_id}",
                f"company:{company_id}",
                f"is_vague:{job_listing.get('is_vague', False)}",
                f"uses_emojis:{job_listing.get('uses_emojis', False)}"
            ])
        except:
            pass
    
    # Create job description generator agent
    description_generator = autogen.AssistantAgent(
        name="job_description_generator",
        llm_config=llm_config,
        system_message="You create detailed job descriptions based on job listing information."
    )
    
    user_proxy = create_user_proxy(name="user_proxy")
    
    # Prepare style instructions based on flags
    style_instructions = ""
    
    if job_listing.get("is_vague", False):
        style_instructions += "Make the description somewhat vague about specific responsibilities and qualifications. "
    else:
        style_instructions += "Make the description very specific and clear about responsibilities and qualifications. "
    
    if job_listing.get("uses_emojis", False):
        style_instructions += "Use emojis in the description to make it more engaging. "
    else:
        style_instructions += "Keep the description professional without using emojis. "
    
    # Calculate salary range text
    min_salary, max_salary = job_listing.get("salary_range", (MIN_SALARY, MAX_SALARY))
    if min_salary == max_salary:
        salary_text = f"{min_salary:,}€"
    else:
        salary_text = f"{min_salary:,}€ - {max_salary:,}€"
    
    # Add emphasis on specific skills
    emphasis_skills = job_listing.get("emphasis_skills", [])
    if emphasis_skills:
        emphasis_text = f"Emphasize the importance of these key skills: {', '.join(emphasis_skills)}. "
    else:
        emphasis_text = ""
    
    # Generate description
    user_proxy.initiate_chat(
        description_generator, 
        message=f"""
        Create a job description for the following position:
        - Job title: {job_listing.get('title', 'Data Scientist')}
        - Company: {job_listing.get('company_name', 'Company')} (in the {job_listing.get('sector', 'Technology')} sector)
        - Required skills: {', '.join(job_listing.get('required_skills', []))}
        - Salary range: {salary_text}
        
        {style_instructions}
        {emphasis_text}
        
        Format the job posting as a professional job advertisement with sections for:
        1. About the company
        2. Position overview
        3. Key responsibilities
        4. Required qualifications
        5. Benefits & compensation
        """
    )
    
    description = description_generator.last_message()["content"].strip()
    
    # Track completion with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            generation_time = time.time() - start_time
            description_length = len(description)
            agentops.add_tags([
                f"job_description_generated:{job_id}",
                f"description_length:{description_length}",
                f"generation_time:{generation_time:.2f}s"
            ])
        except:
            pass
    
    return description