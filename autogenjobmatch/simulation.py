"""
Main simulation logic.
"""
import time
import random
from typing import Dict, Any, List, Optional
import autogen
from .agents import (
    get_llm_config, 
    CandidateAgent, 
    CompanyAgent, 
    create_interviewer_agent,
    create_user_proxy
)
from .data_generators import (
    generate_candidate_profile,
    generate_company_profile,
    generate_job_offer
)
from .monitoring import AgentOpsTracker
from .config import (
    NUM_CANDIDATES, 
    NUM_COMPANIES, 
    GROQ_API_KEY,
    AGENTOPS_API_KEY,
    INTERVIEW_ROUNDS,
    INTERVIEW_STRATEGIES,
    INTERVIEW_ATTITUDES
)

def run_interview(
    candidate: CandidateAgent,
    company: CompanyAgent,
    job_offer: Dict[str, Any],
    llm_config: Dict[str, Any],
    tracker: Optional[AgentOpsTracker] = None
) -> Dict[str, Any]:
    """
    Run an interview between a candidate and company.
    
    Args:
        candidate: Candidate agent
        company: Company agent
        job_offer: Job offer details
        llm_config: LLM configuration
        tracker: AgentOps tracker
        
    Returns:
        Dictionary with interview results
    """
    interview_id = f"interview_{company.id}_{candidate.id}_{int(time.time())}"
    
    # Track interview start
    if tracker:
        tracker.track_event("interview_started", {
            "interview_id": interview_id,
            "candidate_id": candidate.id,
            "company_id": company.id,
            "job_id": job_offer["id"]
        })
    
    # Create interviewer
    interview_style = random.choice(INTERVIEW_STRATEGIES)
    interviewer = create_interviewer_agent(
        company_name=company.profile["name"],
        position=job_offer["title"],
        style=interview_style,
        llm_config=llm_config,
        tracker=tracker
    )
    
    # Create user proxy for orchestrating the conversation
    user_proxy = create_user_proxy(name="interview_coordinator")
    
    # Set up the group chat
    groupchat = autogen.GroupChat(
        agents=[interviewer, candidate.agent, user_proxy],
        messages=[],
        max_round=INTERVIEW_ROUNDS
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config)
    
    # Start the interview
    user_proxy.initiate_chat(
        manager,
        message=f"""
        This is an interview for the {job_offer['title']} position at {company.profile['name']}.
        
        Interviewer, please conduct the interview with the candidate. Ask about their skills and experience
        related to {', '.join(job_offer['required_skills'])}.
        
        Conduct {INTERVIEW_ROUNDS} rounds of questions and responses.
        """
    )
    
    # Extract the conversation
    messages = groupchat.messages
    
    # Track interview completion
    if tracker:
        tracker.track_event("interview_completed", {
            "interview_id": interview_id,
            "candidate_id": candidate.id,
            "company_id": company.id,
            "job_id": job_offer["id"],
            "num_messages": len(messages)
        })
    
    # Return interview results
    return {
        "interview_id": interview_id,
        "candidate_id": candidate.id,
        "company_id": company.id,
        "job_id": job_offer["id"],
        "style": interview_style,
        "messages": messages,
        "completion_time": time.time()
    }

def run_simulation(
    num_candidates: int = NUM_CANDIDATES,
    num_companies: int = NUM_COMPANIES,
    groq_api_key: str = GROQ_API_KEY,
    agentops_api_key: str = AGENTOPS_API_KEY
) -> Dict[str, Any]:
    """
    Run the job matching simulation.
    
    Args:
        num_candidates: Number of candidates
        num_companies: Number of companies
        groq_api_key: Groq API key
        agentops_api_key: AgentOps API key
        
    Returns:
        Dictionary with simulation results
    """
    start_time = time.time()
    
    # Set up monitoring with AgentOps
    tracker = AgentOpsTracker(api_key=agentops_api_key)
    
    # Add initial tags/metrics
    tracker.update_metrics({
        "num_candidates": num_candidates,
        "num_companies": num_companies
    })
    
    # Set up LLM config
    llm_config = get_llm_config(api_key=groq_api_key)
    
    # Generate candidate profiles
    print(f"Generating {num_candidates} candidate profiles...")
    candidates = []
    for i in range(num_candidates):
        profile = generate_candidate_profile(i, llm_config, tracker)
        candidate = CandidateAgent(i, profile, llm_config, tracker)
        candidates.append(candidate)
    
    # Generate company profiles
    print(f"Generating {num_companies} company profiles...")
    companies = []
    for i in range(num_companies):
        profile = generate_company_profile(i, llm_config, tracker)
        company = CompanyAgent(i, profile, llm_config, tracker)
        companies.append(company)
    
    # Generate job offers
    print("Generating job offers...")
    job_offers = []
    for i, company in enumerate(companies):
        job_offer = generate_job_offer(company.profile, i, llm_config, tracker)
        job_offers.append(job_offer)
    tracker.update_metrics({"num_job_offers": len(job_offers)})
    
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
        
        # Apply to top 2 jobs
        energy = candidate.profile["energy"]
        for offer, score in scored_offers[:2]:
            if energy > 0:
                applications.append({
                    "candidate_id": candidate.id,
                    "job_id": offer["id"],
                    "company_id": offer["company_id"],
                    "score": score
                })
                energy -= 1
    
    tracker.update_metrics({"num_applications": len(applications)})
    print(f"Generated {len(applications)} applications")
    
    # Run interviews (for a subset of applications)
    print("Running interviews...")
    interviews = []
    for application in applications[:min(5, len(applications))]:  # Limit for initial testing
        candidate = next(c for c in candidates if c.id == application["candidate_id"])
        company = next(c for c in companies if c.id == application["company_id"])
        job_offer = next(j for j in job_offers if j["id"] == application["job_id"])
        
        interview_result = run_interview(candidate, company, job_offer, llm_config, tracker)
        interviews.append(interview_result)
    
    tracker.update_metrics({"num_interviews": len(interviews)})
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
        
        tracker.track_event("hiring_decision", {
            "interview_id": interview["interview_id"],
            "candidate_id": candidate.id,
            "company_id": company.id,
            "hired": hired
        })
    
    # Count number of hires
    num_hires = sum(1 for d in hiring_decisions if d["hired"])
    tracker.update_metrics({"num_hires": num_hires})
    
    # Prepare results
    results = {
        "simulation_id": tracker.session_id,
        "duration": time.time() - start_time,
        "num_candidates": num_candidates,
        "num_companies": num_companies,
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
                "num_messages": len(i["messages"])
            }
            for i in interviews
        ],
        "hiring_decisions": hiring_decisions,
        "num_hires": num_hires
    }
    
    # End tracking session
    tracker.end_session(success=True, metadata={
        "num_candidates": num_candidates,
        "num_companies": num_companies,
        "num_job_offers": len(job_offers),
        "num_applications": len(applications),
        "num_interviews": len(interviews),
        "num_hires": num_hires,
        "duration": time.time() - start_time
    })
    
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    print(f"Results: {num_hires} hires from {len(interviews)} interviews")
    
    return results