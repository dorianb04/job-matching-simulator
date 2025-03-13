"""
Functions to generate data for the job matching simulation.
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
    LIE_PROBABILITY
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
    Generate a candidate profile.
    
    Args:
        candidate_id: Unique identifier
        llm_config: LLM configuration
        
    Returns:
        Dictionary with candidate profile
    """
    start_time = time.time()
    
    # Track operation with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"generating_candidate:{candidate_id}"])
        except:
            pass
    
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
    
    # Track completion with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            generation_time = time.time() - start_time
            agentops.add_tags([
                f"candidate_generated:{candidate_id}",
                f"generation_time:{generation_time:.2f}s"
            ])
        except:
            pass
    
    return profile

@agentops.record_action('generate_company_profile') if AGENTOPS_AVAILABLE else lambda func: func
def generate_company_profile(
    company_id: int,
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a company profile.
    
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
    
    # Generate random budget
    budget = random.randint(MIN_SALARY, MAX_SALARY)
    
    profile = {
        "id": company_id,
        "name": company_name,
        "sector": sector,
        "budget": budget,
        "required_skills": required_skills
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

@agentops.record_action('generate_job_offer') if AGENTOPS_AVAILABLE else lambda func: func
def generate_job_offer(
    company_profile: Dict[str, Any],
    job_id: int,
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a job offer for a company.
    
    Args:
        company_profile: Company profile
        job_id: Unique job identifier
        llm_config: LLM configuration
        
    Returns:
        Dictionary with job offer details
    """
    start_time = time.time()
    
    # Track operation with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"generating_job_offer:{job_id}",
                f"company:{company_profile['id']}"
            ])
        except:
            pass
    
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
    
    # Determine if job description should be vague or use emojis
    is_vague = random.random() < VAGUE_JOB_DESCRIPTION_PROBABILITY
    uses_emojis = random.random() < EMOJI_PROBABILITY
    
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
    
    # Track completion with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            generation_time = time.time() - start_time
            agentops.add_tags([
                f"job_offer_generated:{job_id}",
                f"job_title:{job_title}",
                f"generation_time:{generation_time:.2f}s"
            ])
        except:
            pass
    
    return job_offer

@agentops.record_action('generate_cv') if AGENTOPS_AVAILABLE else lambda func: func
def generate_cv(
    candidate_profile: Dict[str, Any],
    truth_level: float = 1.0
) -> Dict[str, Any]:
    """
    Generate a CV for a candidate, potentially with exaggerations.
    
    Args:
        candidate_profile: Candidate profile
        truth_level: How truthful to be (1.0 = completely honest)
        
    Returns:
        Dictionary with CV data
    """
    start_time = time.time()
    candidate_id = candidate_profile["id"]
    
    # Track operation with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"generating_cv:{candidate_id}",
                f"truth_level:{truth_level:.2f}"
            ])
        except:
            pass
    
    skills = {}
    
    # Copy skills, potentially with exaggerations
    for skill, level in candidate_profile.get("skills", {}).items():
        if random.random() > truth_level:
            # Exaggerate skill level (but not beyond 10)
            exaggerated_level = min(level + random.randint(1, 3), 10)
            skills[skill] = exaggerated_level
        else:
            skills[skill] = level
    
    # Generate expected salary based on skills and money importance
    avg_skill_level = sum(skills.values()) / len(skills) if skills else 5
    money_importance = candidate_profile.get("money_importance", 0.5)
    
    # Higher skills and money importance result in higher salary expectations
    expected_salary = int(MIN_SALARY + (MAX_SALARY - MIN_SALARY) * 
                         (0.3 + 0.4 * (avg_skill_level / 10) + 0.3 * money_importance))
    
    cv = {
        "candidate_id": candidate_id,
        "skills": skills,
        "truth_level": truth_level,
        "expected_salary": expected_salary,
        "exaggerated_skills": [skill for skill, level in skills.items() 
                              if level > candidate_profile["skills"].get(skill, 0)]
    }
    
    # Track completion with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            generation_time = time.time() - start_time
            agentops.add_tags([
                f"cv_generated:{candidate_id}",
                f"exaggerations:{len(cv['exaggerated_skills'])}",
                f"generation_time:{generation_time:.2f}s"
            ])
        except:
            pass
    
    return cv

@agentops.record_action('generate_job_description') if AGENTOPS_AVAILABLE else lambda func: func
def generate_job_description(
    job_offer: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> str:
    """
    Generate a detailed job description for a job offer.
    
    Args:
        job_offer: Job offer data
        llm_config: LLM configuration
        
    Returns:
        Job description text
    """
    start_time = time.time()
    job_id = job_offer["id"]
    
    # Track operation with AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"generating_job_description:{job_id}",
                f"is_vague:{job_offer.get('is_vague', False)}",
                f"uses_emojis:{job_offer.get('uses_emojis', False)}"
            ])
        except:
            pass
    
    # Create job description generator agent
    description_generator = autogen.AssistantAgent(
        name="job_description_generator",
        llm_config=llm_config,
        system_message="You create detailed job descriptions based on job offer information."
    )
    
    user_proxy = create_user_proxy(name="user_proxy")
    
    # Prepare style instructions based on flags
    style_instructions = ""
    
    if job_offer.get("is_vague", False):
        style_instructions += "Make the description somewhat vague about specific responsibilities and qualifications. "
    else:
        style_instructions += "Make the description very specific and clear about responsibilities and qualifications. "
    
    if job_offer.get("uses_emojis", False):
        style_instructions += "Use emojis in the description to make it more engaging. "
    else:
        style_instructions += "Keep the description professional without using emojis. "
    
    # Calculate salary range text
    min_salary, max_salary = job_offer.get("salary_range", (MIN_SALARY, MAX_SALARY))
    if min_salary == max_salary:
        salary_text = f"${min_salary:,}"
    else:
        salary_text = f"${min_salary:,} - ${max_salary:,}"
    
    # Generate description
    user_proxy.initiate_chat(
        description_generator, 
        message=f"""
        Create a job description for the following position:
        - Job title: {job_offer['title']}
        - Company: {job_offer['company_name']} (in the {job_offer['sector']} sector)
        - Required skills: {', '.join(job_offer['required_skills'])}
        - Salary range: {salary_text}
        
        {style_instructions}
        
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