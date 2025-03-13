"""
Main simulation logic for job matching.
"""
import time
import random
import uuid
from typing import Dict, Any, List, Optional
import autogen

from .config import (
    NUM_CANDIDATES, 
    NUM_COMPANIES, 
    GROQ_API_KEY,
    SECTORS,
    DATA_SCIENCE_SKILLS,
    INITIAL_ENERGY,
    INTERVIEW_STRATEGIES,
    INTERVIEW_ATTITUDES,
    DEFAULT_INTERVIEW_ROUNDS,
    MIN_INTERVIEW_ROUNDS,
    MAX_INTERVIEW_ROUNDS,
    CODE_EXECUTION_CONFIG
)
from .agents import (
    get_llm_config, 
    CandidateAgent, 
    CompanyAgent,
    create_user_proxy, 
    create_interviewer_agent
)
from .data_generators import (
    generate_candidate_profile,
    generate_company_profile,
    generate_job_offer
)

# Try to import AgentOps for tracking (optional)
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False

@agentops.record_action('run_interview') if AGENTOPS_AVAILABLE else lambda func: func
def run_interview(
    candidate: CandidateAgent,
    company: CompanyAgent,
    job_offer: Dict[str, Any],
    llm_config: Dict[str, Any],
    num_rounds: int = DEFAULT_INTERVIEW_ROUNDS
) -> Dict[str, Any]:
    """
    Run an interview between a candidate and company.
    
    Args:
        candidate: Candidate agent
        company: Company agent
        job_offer: Job offer details
        llm_config: LLM configuration
        num_rounds: Number of interview rounds
        
    Returns:
        Dictionary with interview results
    """
    interview_id = f"interview_{company.id}_{candidate.id}_{int(time.time())}"
    
    # Add tag for AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"interview:{interview_id}", 
                f"candidate:{candidate.id}", 
                f"company:{company.id}"
            ])
        except:
            pass
    
    # Create interviewer
    interview_style = random.choice(INTERVIEW_STRATEGIES)
    
    # Create safe names for agents (no spaces)
    safe_company_name = company.profile["name"].replace(" ", "_").replace("-", "")
    safe_position = job_offer["title"].replace(" ", "_").replace("-", "")
    
    # Create interviewer with safe name
    interviewer = create_interviewer_agent(
    company_name=company.profile["name"],
    position=job_offer["title"],
    style=interview_style,
    llm_config=llm_config
)
    
    # Create user proxy for orchestrating the conversation
    user_proxy = create_user_proxy(name="interview_coordinator")
    
    # Set up the group chat
    groupchat = autogen.GroupChat(
        agents=[interviewer, candidate.agent, user_proxy],
        messages=[],
        max_round=num_rounds
    )
    
    # Create the manager with explicit LLM config
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Start the interview
    user_proxy.initiate_chat(
        manager,
        message=f"""
        This is an interview for the {job_offer['title']} position at {company.profile['name']}.
        
        Interviewer, please conduct the interview with the candidate. Ask about their skills and experience
        related to {', '.join(job_offer['required_skills'])}.
        
        Conduct {num_rounds} rounds of questions and responses.
        """
    )
    
    # Extract the conversation
    messages = groupchat.messages
    
    # Add completion tag for AgentOps if available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"interview_completed:{interview_id}",
                f"interview_messages:{len(messages)}"
            ])
        except:
            pass
    
    # Return interview results
    return {
        "interview_id": interview_id,
        "candidate_id": candidate.id,
        "company_id": company.id,
        "job_id": job_offer["id"],
        "style": interview_style,
        "num_rounds": num_rounds,
        "messages": messages,
        "completion_time": time.time()
    }

@agentops.record_action('run_simulation') if AGENTOPS_AVAILABLE else lambda func: func
def run_simulation(
    num_candidates: int = NUM_CANDIDATES,
    num_companies: int = NUM_COMPANIES,
    groq_api_key: str = GROQ_API_KEY,
    interview_rounds: int = DEFAULT_INTERVIEW_ROUNDS
) -> Dict[str, Any]:
    """
    Run the job matching simulation.
    
    Args:
        num_candidates: Number of candidates
        num_companies: Number of companies
        groq_api_key: Groq API key
        interview_rounds: Number of rounds per interview
        
    Returns:
        Dictionary with simulation results
    """
    # Validate interview rounds
    interview_rounds = max(MIN_INTERVIEW_ROUNDS, min(MAX_INTERVIEW_ROUNDS, interview_rounds))
    
    start_time = time.time()
    session_id = str(uuid.uuid4())  # Generate a session ID
    
    # Try to add tags if AgentOps is available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"session:{session_id}",
                f"candidates:{num_candidates}",
                f"companies:{num_companies}",
                f"rounds:{interview_rounds}"
            ])
        except:
            pass
    
    # Set up LLM config
    llm_config = get_llm_config(api_key=groq_api_key)
    
    # Generate candidate profiles
    print(f"Generating {num_candidates} candidate profiles...")
    candidates = []
    for i in range(num_candidates):
        profile = generate_candidate_profile(i, llm_config)
        candidate = CandidateAgent(i, profile, llm_config)
        candidates.append(candidate)
    
    # Generate company profiles
    print(f"Generating {num_companies} company profiles...")
    companies = []
    for i in range(num_companies):
        profile = generate_company_profile(i, llm_config)
        company = CompanyAgent(i, profile, llm_config)
        companies.append(company)
    
    # Generate job offers
    print("Generating job offers...")
    job_offers = []
    for i, company in enumerate(companies):
        job_offer = generate_job_offer(company.profile, i, llm_config)
        job_offers.append(job_offer)
    
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"job_offers:{len(job_offers)}"])
        except:
            pass
    
    # Candidates evaluate job offers
    print("Candidates evaluating job offers...")
    applications = []
    for candidate in candidates:
        # Sort job offers by score
        scored_offers = [
            (offer, candidate.evaluate_job_offer(offer))
            for offer in job_offers
        ]
        scored_offers.sort(key=lambda x: x[1], reverse=True)
        
        # Apply to top 2 jobs (or fewer if candidate has less energy)
        energy = candidate.profile["energy"]
        for offer, score in scored_offers[:min(2, energy)]:
            if energy > 0:
                applications.append({
                    "candidate_id": candidate.id,
                    "job_id": offer["id"],
                    "company_id": offer["company_id"],
                    "score": score
                })
                energy -= 1
    
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"applications:{len(applications)}"])
        except:
            pass
            
    print(f"Generated {len(applications)} applications")
    
    # Run interviews (for a subset of applications)
    print(f"Running interviews with {interview_rounds} rounds each...")
    interviews = []
    max_interviews = min(5, len(applications))  # Limit for initial testing
    
    for i, application in enumerate(applications[:max_interviews]):
        print(f"Running interview {i+1}/{max_interviews}...")
        candidate = next(c for c in candidates if c.id == application["candidate_id"])
        company = next(c for c in companies if c.id == application["company_id"])
        job_offer = next(j for j in job_offers if j["id"] == application["job_id"])
        
        interview_result = run_interview(
            candidate, 
            company, 
            job_offer, 
            llm_config,
            num_rounds=interview_rounds
        )
        interviews.append(interview_result)
    
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"interviews:{len(interviews)}"])
        except:
            pass
            
    print(f"Completed {len(interviews)} interviews")
    
    # Make hiring decisions
    print("Making hiring decisions...")
    hiring_decisions = []
    for interview in interviews:
        company = next(c for c in companies if c.id == interview["company_id"])
        candidate = next(c for c in candidates if c.id == interview["candidate_id"])
        
        # Simple hiring decision (can be expanded)
        hiring_score = random.random()  # Placeholder for a more sophisticated evaluation
        hired = hiring_score > 0.5
        
        decision = {
            "interview_id": interview["interview_id"],
            "candidate_id": candidate.id,
            "company_id": company.id,
            "hired": hired,
            "score": hiring_score
        }
        hiring_decisions.append(decision)
        
        if AGENTOPS_AVAILABLE:
            try:
                agentops.add_tags([
                    f"hiring_decision:{interview['interview_id']}",
                    f"hired:{hired}"
                ])
            except:
                pass
    
    # Count number of hires
    num_hires = sum(1 for d in hiring_decisions if d["hired"])
    
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"num_hires:{num_hires}"])
        except:
            pass
    
    # Prepare results
    results = {
        "simulation_id": session_id,
        "duration": time.time() - start_time,
        "num_candidates": num_candidates,
        "num_companies": num_companies,
        "interview_rounds": interview_rounds,
        "candidates": [c.profile for c in candidates],
        "companies": [c.profile for c in companies],
        "job_offers": job_offers,
        "applications": applications,
        "interviews": [
            {
                "interview_id": i["interview_id"],
                "candidate_id": i["candidate_id"],
                "company_id": i["company_id"],
                "job_id": i["job_id"],
                "num_rounds": i.get("num_rounds", interview_rounds),
                "num_messages": len(i["messages"])
            }
            for i in interviews
        ],
        "hiring_decisions": hiring_decisions,
        "num_hires": num_hires
    }
    
    # Try to add final stats as tags if AgentOps is available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"duration:{time.time() - start_time:.1f}s"
            ])
        except:
            pass
    
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    print(f"Results: {num_hires} hires from {len(interviews)} interviews")
    
    return results