"""
Main simulation logic for job matching with competitive agents.
"""
import time
import random
import uuid
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple
import autogen
from collections import defaultdict
from typing import Union, Literal
from autogen import Agent, GroupChat

from .config import (
    NUM_CANDIDATES, 
    NUM_COMPANIES,
    NUM_SIMULATION_ROUNDS,
    GROQ_API_KEY,
    SECTORS,
    DATA_SCIENCE_SKILLS,
    INITIAL_ENERGY,
    INTERVIEW_STRATEGIES,
    INTERVIEW_ATTITUDES,
    DEFAULT_INTERVIEW_ROUNDS,
    MIN_INTERVIEW_ROUNDS,
    MAX_INTERVIEW_ROUNDS,
    CODE_EXECUTION_CONFIG,
    MAX_APPLICATIONS_PER_CANDIDATE,
    MAX_INTERVIEWS_PER_COMPANY,
    CANDIDATE_REWARD_WEIGHTS,
    COMPANY_REWARD_WEIGHTS,
    DEFAULT_OUTPUT_DIR,
)
from .agents import (
    get_llm_config, 
    CandidateAgent, 
    CompanyAgent,
    create_user_proxy
)
from .data_generators import (
    generate_candidate_profile,
    generate_company_profile,
    generate_job_description
)

# Try to import AgentOps for tracking (optional)
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False


def run_interview(
    candidate: CandidateAgent,
    company: CompanyAgent,
    job_listing: Dict[str, Any],
    interview_process: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a structured interview using a group chat with alternating speakers.
    
    Args:
        candidate: Candidate agent
        company: Company agent
        job_listing: Job listing details
        interview_process: Interview process design
        llm_config: LLM configuration
        
    Returns:
        Dictionary with interview results
    """
    from typing import Union, Literal
    from autogen import Agent, GroupChat
    
    interview_id = f"interview_{company.id}_{candidate.id}_{int(time.time())}"
    
    # Get interview parameters
    interview_style = interview_process.get("style", "Technical")
    interview_attitude = interview_process.get("attitude", "Formal")
    num_rounds = interview_process.get("num_rounds", 3)
    
    print(f"\n------ STARTING INTERVIEW: Company {company.id} interviewing Candidate {candidate.id} ------")
    print(f"Style: {interview_style}, Attitude: {interview_attitude}, Rounds: {num_rounds}")
    
    # Get candidate's resume
    candidate_resume = None
    for app in candidate.application_history:
        if app.get("job_id") == job_listing.get("id"):
            candidate_resume = app.get("resume", {})
            break
    
    # Create interviewer agent
    interviewer = autogen.AssistantAgent(
        name=f"interviewer_{company.id}",
        llm_config=llm_config,
        system_message=f"""
        You are an interviewer for {company.profile.get('name', f'Company {company.id}')}.
        You are interviewing a junior candidate for the {job_listing.get('title', 'Junior Data Scientist')} position.
        
        Your interview style is {interview_style} and your attitude is {interview_attitude}.
        
        You are conducting {num_rounds} rounds of questions and responses. (1 round consists of 1 question and 1 answer)
        
        THE STRUCTURE MUST BE:
        1. You start with a brief greeting and your first question
        2. The candidate will answer
        3. You ask your second question
        4. The candidate will answer
        5. Continue until all rounds are complete
        6. You end with a brief closing statement after the last candidate response
        
        IMPORTANT:
        - Ask exactly ONE question per round
        - Focus on assessing these skills: {', '.join(job_listing.get('required_skills', []))}
        - Keep your questions appropriate for a junior-level candidate with limited experience
        - Pay attention to inconsistencies in the candidate's knowledge vs. claimed skills
        """
    )
    
    # Configure candidate agent with strategy parameters from interview_process
    education = candidate_resume.get('content', {}).get('education', [{}])
    try:
        university = education.get('school_name', 'a French university') if education else 'a French university'
    except:
        university = education[0].get('school_name', 'a French university') if education else 'a French university'
    finally:
        university = "a French university" if not university else university
    
    # Get candidate's strategic parameters
    candidate_style = interview_process.get("strategy", candidate.interview_style)
    candidate_attitude = interview_process.get("attitude", candidate.interview_attitude)
    lyingness = interview_process.get("lyingness", candidate.lyingness)
    skills_to_improve = interview_process.get("skills_to_improve", candidate.skills_to_improve)
    
    # Update the system message with strategy information
    candidate.agent.update_system_message(f"""
    You are a junior-level job candidate who recently graduated from {university}.
    You are interviewing for a {job_listing.get('title', 'Junior Data Scientist')} position.
    
    Your profile:
    - Skills: {json.dumps(candidate.profile.get('skills', {}))}
    - Education: {json.dumps(education)}
    
    Your interview strategy:
    - Style: {candidate_style} (be {'formal and professional' if candidate_style == 'Formal' else 'casual and conversational'})
    - Attitude: {candidate_attitude} (be {'humble and modest' if candidate_attitude == 'Humble' else 'confident and assertive'})
    - Skills you're focusing on improving: {', '.join(skills_to_improve)}
    - Honesty level: {1.0 - lyingness:.1f} on a scale of 0-1
    
    IMPORTANT ABOUT HONESTY:
    - Your lyingness factor is {lyingness:.1f} on a scale of 0-1
    - With lyingness = 0.0: Be completely honest about your skills and experience
    - With lyingness = 0.5: Moderately exaggerate your achievements and skills
    - With lyingness = 1.0: Fabricate skills and experience substantially
    
    When discussing skills where you've exaggerated or lied:
    - Show appropriate hesitation when asked detailed questions
    - If your true skill level in something is low but you claimed high expertise, you'll struggle to answer technical questions
    - The more you've exaggerated a skill, the less confident your answers should be about it
    - If directly questioned about inconsistencies, either deflect or admit to "overstating"
    
    Answer the interviewer's questions truthfully based on your limited professional experience.
    Be professional but show enthusiasm to learn.
    Your knowledge should be consistent with a recent graduate.
    DO NOT exaggerate your experience or claim expertise you don't have as a junior candidate beyond the level determined by your lyingness factor.
    """)
    
    # Print the candidate's resume for context
    if candidate_resume:
        print("\nCANDIDATE RESUME:")
        print(json.dumps(candidate_resume.get("content", {}), indent=2))
        print("\nBEGINNING INTERVIEW...\n")
    
    # Create a user proxy for the interview coordinator
    coordinator = create_user_proxy(name="interview_coordinator")
    
    # Define custom speaker selection function for alternating between interviewer and candidate
    def custom_speaker_selection_func(
        last_speaker: Agent, 
        groupchat: GroupChat
    ) -> Union[Agent, Literal['auto', 'manual', 'random', 'round_robin'], None]:
        """Custom speaker selection function that alternates between interviewer and candidate."""
        # If the last speaker was the coordinator, the interviewer speaks first
        if last_speaker.name == coordinator.name:
            return interviewer
        
        # If the last speaker was the interviewer, it's candidate's turn
        if last_speaker.name == interviewer.name:
            return candidate.agent
        
        # If the last speaker was the candidate, it's interviewer's turn
        if last_speaker.name == candidate.agent.name:
            return interviewer
        
        # Fallback - should not happen
        return "auto"
    
    # Create the group chat
    groupchat = autogen.GroupChat(
        agents=[interviewer, candidate.agent, coordinator],
        messages=[],
        max_round=num_rounds * 2 + 1,  # Intro + (question+answer)*rounds + closing
        speaker_selection_method=custom_speaker_selection_func
    )
    
    # Create the manager
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Start the interview
    initial_message = f"""
    Begin the interview for the {job_listing.get('title', 'Junior Data Scientist')} position.
    
    The candidate's resume shows:
    - Education: {json.dumps(education)}
    - Skills: {json.dumps(candidate_resume.get('content', {}).get('skills', {}))}
    
    Interviewer, please start with a greeting and your first question for the candidate.
    """
    
    coordinator.initiate_chat(manager, message=initial_message)
    
    # Extract messages from the group chat
    all_messages = []
    for msg in groupchat.messages:
        if msg["role"] == "user" and msg["name"] == coordinator.name:
            continue  # Skip coordinator messages
            
        all_messages.append({
            "name": msg["name"],
            "content": msg["content"]
        })
    
    print(f"\n------ INTERVIEW COMPLETED: {len(all_messages)} messages exchanged ------\n")
    
    # Return interview results
    return {
        "interview_id": interview_id,
        "candidate_id": candidate.id,
        "company_id": company.id,
        "job_id": job_listing.get("id"),
        "style": interview_style,
        "attitude": interview_attitude,
        "num_rounds": num_rounds,
        "messages": all_messages,
        "completion_time": time.time()
    }


def calculate_candidate_reward(candidate: CandidateAgent, outcomes: Dict[str, Any]) -> float:
    """
    Calculate reward for a candidate agent based on French salary standards.
    
    Args:
        candidate: Candidate agent
        outcomes: Outcome data
        
    Returns:
        Reward value (0-1)
    """
    weights = CANDIDATE_REWARD_WEIGHTS
    reward = 0.0
    
    # Get candidate-specific outcomes if using enhanced structure
    if "candidates" in outcomes and candidate.id in outcomes["candidates"]:
        candidate_outcomes = outcomes["candidates"][candidate.id]
        
        # If caught lying, severe penalty regardless of other outcomes
        if candidate_outcomes.get("lying_detected", False):
            return 0.05
        
        # If no offer accepted, reward based on progress
        accepted_offer = candidate_outcomes.get("accepted_offer")
        if not accepted_offer:
            if candidate_outcomes.get("offers_count", 0) > 0:
                return 0.3  # Got offer but declined
            elif candidate_outcomes.get("interviews_count", 0) > 0:
                return 0.2  # Got interview but no offer
            elif len(candidate_outcomes.get("shortlisted_for", [])) > 0:
                return 0.1  # Got shortlisted but no interview
            else:
                return 0.05  # No progress
    else:
        # Legacy/simple outcome structure
        # If no offer accepted, minimal reward
        if not outcomes.get("offer_accepted", False):
            # Check if detected lying
            if outcomes.get("detected_lying", False):
                return 0.05
                
            # Check progress level
            if outcomes.get("received_offer", False):
                return 0.3  # Got offer but declined
            elif outcomes.get("interviewed", False):
                return 0.2  # Got interview but no offer
            elif outcomes.get("application_accepted", False):
                return 0.1  # Got shortlisted but no interview
            else:
                return 0.05  # No progress
    
    # Get accepted offer details (either from enhanced or legacy structure)
    accepted_offer = None
    if "candidates" in outcomes and candidate.id in outcomes["candidates"]:
        accepted_offer = outcomes["candidates"][candidate.id].get("accepted_offer")
    else:
        accepted_offer = outcomes.get("accepted_offer", {})
    
    # Calculate salary reward (normalized to French standards)
    salary = accepted_offer.get("salary", 0) if accepted_offer else 0
    if not salary and "accepted_offer" in outcomes:
        salary = outcomes["accepted_offer"].get("salary", 0)
    
    salary_score = min(salary, 45000) / 45000  # Using French junior salary scale
    
    # Calculate sector alignment
    job_id = None
    company_id = None
    
    if accepted_offer:
        job_id = accepted_offer.get("job_id")
        company_id = accepted_offer.get("company_id")
    
    sector = None
    for app in candidate.application_history:
        if (job_id and app.get("job_id") == job_id) or (app.get("job_id") == outcomes.get("job_id")):
            sector = app.get("job_listing", {}).get("sector")
            break
    
    sector_score = 0.5  # Default
    if sector:
        motivation = candidate.profile.get("motivation", {}).get(sector, 5) / 10
        sector_score = motivation
    
    # Calculate skill utilization
    skill_score = 0.5  # Default
    required_skills = []
    
    for app in candidate.application_history:
        if (job_id and app.get("job_id") == job_id) or (app.get("job_id") == outcomes.get("job_id")):
            required_skills = app.get("job_listing", {}).get("required_skills", [])
            break
    
    if required_skills:
        candidate_skills = candidate.profile.get("skills", {})
        skill_score = sum(candidate_skills.get(skill, 0) for skill in required_skills) / (len(required_skills) * 10)
    
    # Combine using weights
    reward = (
        weights["salary"] * salary_score +
        weights["sector_alignment"] * sector_score +
        weights["skill_utilization"] * skill_score
    )
    
    return reward


def calculate_company_reward(company: CompanyAgent, outcomes: Dict[str, Any]) -> float:
    """
    Calculate reward for a company agent based on French hiring standards.
    
    Args:
        company: Company agent
        outcomes: Outcome data
        
    Returns:
        Reward value (0-1)
    """
    weights = COMPANY_REWARD_WEIGHTS
    reward = 0.0
    
    # Check for company-specific outcomes in enhanced structure
    if "companies" in outcomes and company.id in outcomes["companies"]:
        company_outcomes = outcomes["companies"][company.id]
        
        # Enhanced penalties for companies that don't hire
        hired_candidate_id = company_outcomes.get("hired")
        if not hired_candidate_id:
            if company_outcomes.get("all_candidates_refused", False):
                # Severe penalty for being too selective
                return 0.05
            elif company_outcomes.get("no_qualified_candidates", False):
                # Penalty for failing to attract qualified candidates
                return 0.08
            elif len(company_outcomes.get("applicants", [])) == 0:
                # Penalty for failing to attract any candidates
                return 0.07
            else:
                # Default penalty for other reasons
                return 0.1
    else:
        # Legacy/simple outcome structure
        # If no candidate hired, minimal reward
        if not outcomes.get("candidate_hired", False):
            if outcomes.get("all_candidates_refused", False):
                return 0.05  # Penalty for being too selective
            elif outcomes.get("no_qualified_candidates", False):
                return 0.08  # No qualified candidates
            elif outcomes.get("no_applicants", False):
                return 0.07  # No applicants
            else:
                return 0.1  # Default penalty
    
    # Get hire details - first try enhanced structure
    hired_candidate_id = None
    job_id = None
    
    if "companies" in outcomes and company.id in outcomes["companies"]:
        hired_candidate_id = outcomes["companies"][company.id].get("hired")
        
        # Find job ID from offer history
        for offer in company.offer_history:
            if isinstance(offer, dict) and offer.get("candidate_id") == hired_candidate_id:
                job_id = offer.get("job_id")
                break
    
    # Get candidate skills and job requirements
    candidate_skills = {}
    required_skills = []
    salary = 0
    sector_motivation = 5  # Default on scale of 1-10
    
    # Try to get from hire_details if available
    if "hire_details" in outcomes:
        hire_details = outcomes.get("hire_details", {})
        candidate_skills = hire_details.get("candidate_skills", {})
        required_skills = hire_details.get("required_skills", [])
        salary = hire_details.get("salary", 0)
        sector_motivation = hire_details.get("sector_motivation", 5)
    else:
        # Try to find details from company history
        for job in company.job_listing_history:
            if job.get("id") == job_id:
                required_skills = job.get("required_skills", [])
                break
                
        # Find salary from offer history
        for offer in company.offer_history:
            if isinstance(offer, dict) and offer.get("job_id") == job_id:
                salary = offer.get("salary", 0)
                break
                
        # Use default candidate skills if not available
        if not candidate_skills and required_skills:
            candidate_skills = {skill: 5 for skill in required_skills}  # Default mid-level
    
    # Calculate skill match
    skill_match = 0.5  # Default
    if required_skills and candidate_skills:
        skill_match = sum(candidate_skills.get(skill, 0) for skill in required_skills) / (len(required_skills) * 10)
    
    # Calculate budget efficiency
    budget = company.profile.get("budget", 38000)  # French junior position budget
    
    budget_efficiency = 0.5  # Default
    if budget > 0 and salary > 0:
        # Higher score for lower salary (more budget efficient)
        budget_efficiency = 1.0 - (salary / (budget * 1.5))
        budget_efficiency = max(0.0, min(1.0, budget_efficiency))
    
    # Calculate estimated retention probability
    # Based on skill match and sector alignment
    retention_score = 0.5  # Default
    sector_motivation_normalized = sector_motivation / 10  # Normalize to 0-1
    
    retention_score = 0.5 * skill_match + 0.5 * sector_motivation_normalized
    
    # Combine using weights
    reward = (
        weights["skill_match"] * skill_match +
        weights["budget_efficiency"] * budget_efficiency +
        weights["retention_probability"] * retention_score
    )
    
    return reward

def format_value(val):
    if isinstance(val, (float, int)):
        return f"{val:.2f}"
    else:
        return str(val)

def run_simulation_round(
    candidates: List[CandidateAgent],
    companies: List[CompanyAgent],
    llm_config: Dict[str, Any],
    round_num: int
) -> Dict[str, Any]:
    """
    Run a single round of the simulation.
    
    Args:
        candidates: List of candidate agents
        companies: List of company agents
        llm_config: LLM configuration
        round_num: Simulation round number
        
    Returns:
        Dictionary with round results
    """
    start_time = time.time()
    round_id = f"round_{round_num}_{int(start_time)}"
    
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([f"simulation_round:{round_num}"])
        except:
            pass
    
    # Track round statistics
    stats = {
        "round_id": round_id,
        "round_num": round_num,
        "candidates_hired": 0,
        "total_applications": 0,
        "total_interviews": 0,
        "total_offers": 0,
        "rejected_offers": 0,
        "candidate_rewards": {},
        "company_rewards": {},
        "avg_candidate_reward": 0.0,
        "avg_company_reward": 0.0
    }

    # Enhanced outcome tracking
    outcomes = {
        "candidates": {c.id: {
            "applied_to_jobs": [],
            "shortlisted_for": [],
            "interviewed_for": [],
            "received_offers_from": [],
            "accepted_offer": None,
            "rejected_offers": [],
            "lying_detected": False,
            "applications_count": 0,
            "interviews_count": 0,
            "offers_count": 0,
            "skills_improved": [],
            "strategic_changes": {}
        } for c in candidates},
        "companies": {c.id: {
            "applicants": [],
            "shortlisted": [],
            "interviewed": [],
            "offers_extended": [],
            "hired": None,
            "rejected_candidates": [],
            "detected_lying": [],
            "standards_adjusted": False,
            "budget_adjusted": False,
            "all_candidates_refused": False,
            "no_qualified_candidates": False,
            "strategic_changes": {}
        } for c in companies},
        "round_stats": stats  # Reference to the existing stats dictionary
    }
    
    # Step 1: Companies publish job listings (each company has exactly one position)
    print("Step 1: Companies publishing job listings...")
    job_listings = []
    for company in companies:
        # Each company creates exactly one job listing
        job_listing = company.create_job_listing()
        
        # Ensure it's a junior position
        job_listing["title"] = f"Junior {job_listing.get('title', 'Data Scientist')}" if "Junior" not in job_listing.get("title", "") else job_listing.get("title")
        
        # Set French salary ranges (30-45K EUR for junior positions)
        min_salary = random.randint(30000, 35000)
        max_salary = random.randint(38000, 45000)
        job_listing["salary_range"] = (min_salary, max_salary)
        
        # Generate detailed job description
        description = generate_job_description(job_listing, llm_config)
        job_listing["description"] = description
        
        job_listings.append(job_listing)
        print(f"  - {company.profile['name']} posted: {job_listing.get('title')}")
    
    # Step 2: Candidates evaluate job listings and select which to apply for
    print("\nStep 2: Candidates selecting jobs and creating applications...")
    applications = []
    for candidate in candidates:
        # Use LLM to select jobs to apply for
        selected_jobs = candidate.evaluate_job_listings(job_listings)

        # Track in outcomes
        outcomes["candidates"][candidate.id]["applications_count"] = len(selected_jobs)
        
        # Create applications for selected jobs
        for job in selected_jobs:
            # Create CV/resume
            resume = candidate.create_cv(truth_level=candidate.truth_level)
            resume["company_id"] = job["company_id"]
            resume["job_id"] = job["id"]
            
            # Ensure the expected salary is in French range
            resume["expected_salary"] = random.randint(32000, 42000)
            
            # Add application
            application = {
                "candidate_id": candidate.id,
                "job_id": job["id"],
                "company_id": job["company_id"],
                "resume": resume,
                "date": time.time()
            }
            
            applications.append(application)
            
            # Add to candidate's history
            candidate.application_history.append({
                "job_id": job["id"],
                "company_id": job["company_id"],
                "job_listing": job,
                "resume": resume,
                "date": time.time()
            })
            
            # Update candidate stats
            candidate.profile["energy"] = candidate.profile.get("energy", INITIAL_ENERGY) - 1
            candidate.profile["total_applications"] = candidate.profile.get("total_applications", 0) + 1

            # Track in outcomes
            outcomes["candidates"][candidate.id]["applied_to_jobs"].append(job["id"])
            outcomes["companies"][job["company_id"]]["applicants"].append(candidate.id)
        
        print(f"  - Candidate {candidate.id} applied to {len(selected_jobs)} jobs")
    
    stats["total_applications"] = len(applications)
    
    # Step 3: Companies shortlist candidates
    print("\nStep 3: Companies shortlisting candidates...")
    shortlists = {}
    for company in companies:
        # Get applications for this company
        company_applications = [a for a in applications if a["company_id"] == company.id]
        
        # Use LLM to shortlist candidates
        shortlist = company.shortlist_candidates(
            company_applications, 
            num_slots=min(len(company_applications), MAX_INTERVIEWS_PER_COMPANY)
        )
        
        shortlists[company.id] = shortlist
        
        # Record applications in company history
        for app in company_applications:
            company.application_history.append({
                "application": app,
                "shortlisted": app in shortlist
            })
            
            # Track in outcomes
            candidate_id = app.get("candidate_id")
            job_id = app.get("job_id")
            
            if app in shortlist:
                outcomes["candidates"][candidate_id]["shortlisted_for"].append(job_id)
                outcomes["companies"][company.id]["shortlisted"].append(candidate_id)
        
        print(f"  - {company.profile['name']} shortlisted {len(shortlist)}/{len(company_applications)} candidates")

        # Track if no candidates were qualified
        if len(company_applications) > 0 and len(shortlist) == 0:
            outcomes["companies"][company.id]["no_qualified_candidates"] = True
    
    # Step 4: Run interviews
    print("\nStep 4: Conducting interviews...")
    interviews = []
    for company in companies:
        shortlist = shortlists.get(company.id, [])
        
        for application in shortlist:
            candidate_id = application.get("candidate_id")
            job_id = application.get("job_id")
            
            # Find the candidate
            candidate = next((c for c in candidates if c.id == candidate_id), None)
            if not candidate:
                continue
                
            # Find the job listing
            job_listing = next((j for j in job_listings if j["id"] == job_id), None)
            if not job_listing:
                continue
                
            # Create interview process
            interview_process = company.design_interview_process(application)
            
            # Run the interview
            print(f"  - {company.profile['name']} interviewing Candidate {candidate_id}")
            interview_result = run_interview(
                candidate=candidate,
                company=company,
                job_listing=job_listing,
                interview_process=interview_process,
                llm_config=llm_config
            )
            
            # Evaluate the interview and store score
            evaluation = company.evaluate_interview(interview_result)
            lying_detected = evaluation.get("lying_detected", False)
            
            company.interview_history.append({
                "interview_id": interview_result["interview_id"],
                "candidate_id": candidate_id,
                "score": evaluation["score"],
                "lying_detected": lying_detected
            })
            
            # Track in outcomes
            outcomes["candidates"][candidate_id]["interviewed_for"].append(job_id)
            outcomes["candidates"][candidate_id]["interviews_count"] += 1
            outcomes["companies"][company.id]["interviewed"].append(candidate_id)

            if lying_detected:
                outcomes["candidates"][candidate_id]["lying_detected"] = True
                outcomes["companies"][company.id]["detected_lying"].append(candidate_id)
                print(f"  - Company {company.id} detected dishonesty from Candidate {candidate_id}!")
            
            interviews.append(interview_result)
    
    stats["total_interviews"] = len(interviews)
    
    # Step 5: Companies make hiring decisions and offers
    # Each company can make at most one offer (for their single position)
    print("\nStep 5: Companies making hiring decisions...")
    offers = []
    for company in companies:
        # Get interviews for this company
        company_interviews = [i for i in interviews if i["company_id"] == company.id]
        
        # Skip if no interviews conducted
        if not company_interviews:
            continue
            
        # Rank candidates by interview score
        ranked_interviews = []
        for interview in company_interviews:
            # Find score in history
            score = 0.0
            for record in company.interview_history:
                if record.get("interview_id") == interview["interview_id"]:
                    score = record.get("score", 0.0)
                    break
            ranked_interviews.append((interview, score))
        
        # Sort by score (descending)
        ranked_interviews.sort(key=lambda x: x[1], reverse=True)
        
        # Make offer only to the top candidate if they meet the hiring standards
        print(f"  - {company.profile['name']} ranked candidates: {[i[0]['candidate_id'] for i in ranked_interviews]}")
        print(f"  - {company.profile['name']} hiring standards: {company.hiring_standards}")
        print(f"  - {company.profile['name']} interview scores: {[i[1] for i in ranked_interviews]}")
        if ranked_interviews and ranked_interviews[0][1] >= company.hiring_standards:
            print(f"  - {company.profile['name']} making offer to Candidate {ranked_interviews[0][0]['candidate_id']}")
            top_interview = ranked_interviews[0][0]
            candidate_id = top_interview["candidate_id"]
            job_id = top_interview["job_id"]
            
            # Make offer
            offer = company.make_job_offer(candidate_id, job_id)
            if offer:
                # Ensure salary is in French range
                job_listing = next((j for j in job_listings if j["id"] == job_id), None)
                if job_listing:
                    min_salary, max_salary = job_listing.get("salary_range", (30000, 45000))
                    offer["salary"] = int(min_salary + (max_salary - min_salary) * ranked_interviews[0][1])
                
                offers.append(offer)
                
                # Track in outcomes
                outcomes["candidates"][candidate_id]["received_offers_from"].append(company.id)
                outcomes["candidates"][candidate_id]["offers_count"] += 1
                outcomes["companies"][company.id]["offers_extended"].append(candidate_id)
                
                print(f"  - {company.profile['name']} made offer to Candidate {candidate_id} with €{offer['salary']} salary")
        else:
            # Track that the company refused all candidates
            if len(company_interviews) > 0:
                outcomes["companies"][company.id]["all_candidates_refused"] = True
                print(f"  - {company.profile['name']} refused all interviewed candidates")
    
    stats["total_offers"] = len(offers)
    
    # Step 6: Candidates evaluate offers
    print("\nStep 6: Candidates evaluating job offers...")
    candidate_decisions = []
    for candidate in candidates:
        # Get offers for this candidate
        candidate_offers = [o for o in offers if o["candidate_id"] == candidate.id]
        
        if candidate_offers:
            # Update candidate stats
            candidate.profile["offers_received"] = candidate.profile.get("offers_received", 0) + len(candidate_offers)
            
            # Add to history
            for offer in candidate_offers:
                candidate.offer_history.append({
                    "offer": offer,
                    "date": time.time(),
                    "status": "pending"
                })
            
            # Use LLM to rank and decide on offers
            ranked_offers = candidate.rank_job_offers(candidate_offers)
            
            # Process decisions
            for offer in ranked_offers:
                if offer.get("accepted", False):
                    decision = {
                        "candidate_id": candidate.id,
                        "job_id": offer.get("job_id"),
                        "company_id": offer.get("company_id"),
                        "accepted": True,
                        "reason": "Offer meets candidate's expectations"
                    }
                    candidate_decisions.append(decision)
                    
                    # Track in outcomes
                    outcomes["candidates"][candidate.id]["accepted_offer"] = {
                        "company_id": offer.get("company_id"),
                        "job_id": offer.get("job_id"),
                        "salary": offer.get("salary")
                    }
                    outcomes["companies"][offer.get("company_id")]["hired"] = candidate.id
                    
                    print(f"  - Candidate {candidate.id} ACCEPTED offer from Company {offer.get('company_id')} for €{offer.get('salary')}")
                    break  # Only accept one offer
                else:
                    decision = {
                        "candidate_id": candidate.id,
                        "job_id": offer.get("job_id"),
                        "company_id": offer.get("company_id"),
                        "accepted": False,
                        "reason": "Better opportunities available"
                    }
                    candidate_decisions.append(decision)
                    
                    # Track rejected offer
                    outcomes["candidates"][candidate.id]["rejected_offers"].append(offer.get("company_id"))
                    
                    print(f"  - Candidate {candidate.id} REJECTED offer from Company {offer.get('company_id')} for €{offer.get('salary')}")
                    stats["rejected_offers"] += 1
    
    # Step 7: Process hiring decisions
    print("\nStep 7: Processing final hiring decisions...")
    hiring_results = []
    for decision in candidate_decisions:
        candidate_id = decision.get("candidate_id")
        company_id = decision.get("company_id")
        
        # Find the company
        company = next((c for c in companies if c.id == company_id), None)
        if not company:
            continue
            
        # Process the decision
        result = company.process_candidate_decision(decision)
        hiring_results.append(result)
        
        if result.get("accepted", False):
            # Update statistics
            stats["candidates_hired"] += 1
            
            # Update company profile
            company.profile["applicants_hired"] = company.profile.get("applicants_hired", 0) + 1
            
            print(f"  - Company {company_id} hired Candidate {candidate_id}!")
    
    # Step 8: Calculate rewards and adjust strategies
    print("\nStep 8: Calculating rewards and adjusting strategies...")
    
    # Calculate rewards for candidates
    for candidate in candidates:
        # Get candidate-specific outcomes
        candidate_outcomes = outcomes["candidates"][candidate.id]
        
        # Check if hired
        hired = candidate_outcomes["accepted_offer"] is not None
        
        if hired:
            # Get the accepted offer details
            accepted_offer = None
            company_id = candidate_outcomes["accepted_offer"]["company_id"]
            job_id = candidate_outcomes["accepted_offer"]["job_id"]
            
            for offer in offers:
                if offer.get("candidate_id") == candidate.id and offer.get("company_id") == company_id:
                    accepted_offer = offer
                    break
            
            # Calculate reward using the enhanced outcomes
            reward = calculate_candidate_reward(candidate, outcomes)
            stats["candidate_rewards"][candidate.id] = reward
            
            print(f"  - Candidate {candidate.id} reward: {reward:.2f}")
            
            # Get successful skills for strategy adjustment
            successful_skills = []
            for app in candidate.application_history:
                if app.get("company_id") == company_id and app.get("job_id") == job_id:
                    job_listing = app.get("job_listing", {})
                    successful_skills = job_listing.get("required_skills", [])
                    break
            
            # Add to candidate outcomes
            candidate_outcomes["successful_skills"] = successful_skills
            
            # Adjust strategy with enhanced outcomes
            strategic_changes = candidate.adjust_strategy(reward, candidate_outcomes)
            
            # Record strategic changes
            if strategic_changes:
                candidate_outcomes["strategic_changes"] = strategic_changes
                
                # Log significant changes
                for param, change in strategic_changes.items():
                    if isinstance(change, dict) and "before" in change and "after" in change:
                        print(f"    * {param}: {format_value(change['before'])} → {format_value(change['after'])} ({change['reason']})")
                    elif param == "skills_improved" and isinstance(change, list):
                        print(f"    * Improved skills: {', '.join(change)}")
        else:
            # Calculate reward based on progress
            reward = calculate_candidate_reward(candidate, outcomes)
            stats["candidate_rewards"][candidate.id] = reward
            
            print(f"  - Candidate {candidate.id} reward: {reward:.2f}")
            
            # Check if lying was detected in any interview
            lying_detected = candidate_outcomes["lying_detected"]
            
            # Adjust strategy with enhanced outcomes
            strategic_changes = candidate.adjust_strategy(reward, candidate_outcomes)
            
            # Record strategic changes
            if strategic_changes:
                candidate_outcomes["strategic_changes"] = strategic_changes
                
                # Log significant changes
                for param, change in strategic_changes.items():
                    if isinstance(change, dict) and "before" in change and "after" in change:
                        print(f"    * {param}: {format_value(change['before'])} → {format_value(change['after'])} ({change['reason']})")
                    elif param == "skills_improved" and isinstance(change, list):
                        print(f"    * Improved skills: {', '.join(change)}")
    
    # Calculate rewards for companies
    for company in companies:
        # Get company-specific outcomes
        company_outcomes = outcomes["companies"][company.id]
        
        # Check if hired
        hired = company_outcomes["hired"] is not None
        
        if hired:
            # Get hire details
            candidate_id = company_outcomes["hired"]
            job_id = None
            
            # Find job_id from offers
            for offer in offers:
                if offer.get("company_id") == company.id and offer.get("candidate_id") == candidate_id:
                    job_id = offer.get("job_id")
                    break
            
            candidate = next((c for c in candidates if c.id == candidate_id), None)
            job_listing = next((j for j in job_listings if j["id"] == job_id), None)
            
            if candidate and job_listing:
                # Find the offer
                salary = 0
                for offer in offers:
                    if offer.get("company_id") == company.id and offer.get("candidate_id") == candidate_id:
                        salary = offer.get("salary", 0)
                        break
                
                # Gather hire details
                hire_details = {
                    "candidate_id": candidate_id,
                    "job_id": job_id,
                    "required_skills": job_listing.get("required_skills", []),
                    "candidate_skills": candidate.profile.get("skills", {}),
                    "salary": salary,
                    "sector_motivation": candidate.profile.get("motivation", {}).get(company.profile.get("sector"), 5)
                }
                
                # Add to company outcomes
                company_outcomes["hire_details"] = hire_details
                
                # Calculate reward using enhanced outcomes
                reward = calculate_company_reward(company, outcomes)
                stats["company_rewards"][company.id] = reward
                
                print(f"  - Company {company.id} reward: {reward:.2f}")
                
                # Get successful skills
                successful_skills = job_listing.get("required_skills", [])
                below_budget = salary < company.profile.get("budget", 38000)
                
                # Add to company outcomes
                company_outcomes["successful_skills"] = successful_skills
                company_outcomes["below_budget"] = below_budget
                
                # Adjust strategy with enhanced outcomes
                strategic_changes = company.adjust_strategy(reward, company_outcomes)
                
                # Record strategic changes
                if strategic_changes:
                    company_outcomes["strategic_changes"] = strategic_changes
                    
                    # Log significant changes
                    for param, change in strategic_changes.items():
                        if isinstance(change, dict) and "before" in change and "after" in change:
                            print(f"    * {param}: {format_value(change['before'])} → {format_value(change['after'])} ({change['reason']})")
        else:
            # Set special outcomes flags for companies that didn't hire
            all_candidates_refused = company_outcomes.get("all_candidates_refused", False)
            no_qualified_candidates = company_outcomes.get("no_qualified_candidates", False)
            no_applicants = len(company_outcomes.get("applicants", [])) == 0
            
            # Calculate reward using enhanced outcomes
            reward = calculate_company_reward(company, outcomes)
            stats["company_rewards"][company.id] = reward
            
            print(f"  - Company {company.id} reward: {reward:.2f}")
            
            # Check if salary was a constraint
            budget_constraint = False
            for app in applications:
                if app.get("company_id") == company.id:
                    expected_salary = app.get("resume", {}).get("expected_salary", 0)
                    if expected_salary > company.profile.get("budget", 38000):
                        budget_constraint = True
                        break
            
            # Add to company outcomes
            company_outcomes["budget_constraint"] = budget_constraint
            company_outcomes["candidate_shortage"] = no_applicants or no_qualified_candidates
            
            # Adjust strategy with enhanced outcomes
            strategic_changes = company.adjust_strategy(reward, company_outcomes)
            
            # Record strategic changes
            if strategic_changes:
                company_outcomes["strategic_changes"] = strategic_changes
                
                # Log significant changes
                for param, change in strategic_changes.items():
                    if isinstance(change, dict) and "before" in change and "after" in change:
                        print(f"    * {param}: {format_value(change['before'])} → {format_value(change['after'])} ({change['reason']})")
    
    # Calculate average rewards
    if stats["candidate_rewards"]:
        stats["avg_candidate_reward"] = sum(stats["candidate_rewards"].values()) / len(stats["candidate_rewards"])
    
    if stats["company_rewards"]:
        stats["avg_company_reward"] = sum(stats["company_rewards"].values()) / len(stats["company_rewards"])
    
    # Store strategy changes in stats
    if "strategy_changes" not in stats:
        stats["strategy_changes"] = {
            "candidates": {},
            "companies": {}
        }
    
    # Store candidate strategy changes
    for candidate in candidates:
        candidate_id = candidate.id
        if candidate_id not in stats["strategy_changes"]["candidates"]:
            stats["strategy_changes"]["candidates"][candidate_id] = []
        
        # Record current strategic state
        current_state = {
            "round": round_num,
            "lyingness": candidate.lyingness,
            "truth_level": candidate.truth_level,
            "interview_style": candidate.interview_style,
            "interview_attitude": candidate.interview_attitude,
            "skills_to_improve": candidate.skills_to_improve.copy() if hasattr(candidate, "skills_to_improve") else []
        }
        stats["strategy_changes"]["candidates"][candidate_id].append(current_state)
    
    # Store company strategy changes
    for company in companies:
        company_id = company.id
        if company_id not in stats["strategy_changes"]["companies"]:
            stats["strategy_changes"]["companies"][company_id] = []
        
        current_state = {
            "round": round_num,
            "hiring_standards": company.hiring_standards,
            "budget_flexibility": company.budget_flexibility,
            "skill_emphasis": company.skill_emphasis.copy() if hasattr(company, "skill_emphasis") else []
        }
        stats["strategy_changes"]["companies"][company_id].append(current_state)
    
    # Add final stats
    stats["duration"] = time.time() - start_time
    print(f"\nRound {round_num+1} completed in {stats['duration']:.2f} seconds")
    print(f"Hired {stats['candidates_hired']} candidates out of {len(candidates)}")
    
    return stats


def run_simulation(
    num_candidates: int = NUM_CANDIDATES,
    num_companies: int = NUM_COMPANIES,
    num_rounds: int = NUM_SIMULATION_ROUNDS,
    groq_api_key: str = GROQ_API_KEY,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    candidate_strategies: List[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the job matching simulation with strategic candidates.
    
    Args:
        num_candidates: Number of candidates
        num_companies: Number of companies
        num_rounds: Number of simulation rounds
        groq_api_key: Groq API key
        output_dir: Directory for saving results
        candidate_strategies: List of strategy configurations for candidates
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary with simulation results
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())  # Generate a session ID
    
    # Try to add tags if AgentOps is available
    if AGENTOPS_AVAILABLE:
        try:
            agentops.add_tags([
                f"session:{session_id}",
                f"candidates:{num_candidates}",
                f"companies:{num_companies}",
                f"rounds:{num_rounds}"
            ])
        except:
            pass
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up LLM config
    llm_config = get_llm_config(api_key=groq_api_key, verbose=verbose)
    
    # Enhanced tracking structures for plotting
    temporal_data = {
        "rounds": [],
        "candidates": {i: {"skills": [], "strategies": [], "rewards": [], "interviews": [], "offers": []} for i in range(num_candidates)},
        "companies": {i: {"standards": [], "flexibility": [], "rewards": [], "applicants": [], "hires": []} for i in range(num_companies)}
    }
    
    correlation_data = {
        "lyingness_vs_success": [],
        "standards_vs_success": [],
        "budget_vs_salary": [],
        "skills_vs_interviews": []
    }
    
    network_data = {
        "applications": [],
        "interviews": [],
        "offers": [],
        "hires": []
    }
    
    # Generate candidate profiles
    print(f"Generating {num_candidates} junior candidates from French universities...")
    candidates = []
    
    # If strategies are provided, use them
    if candidate_strategies and len(candidate_strategies) > 0:
        # Use only enough candidates to match the strategies
        num_candidates = min(num_candidates, len(candidate_strategies))
        
        for i in range(num_candidates):
            # Generate basic profile
            profile = generate_candidate_profile(i, llm_config)
            
            # Apply strategy configuration
            strategy = candidate_strategies[i]
            profile.update({
                "interview_style": strategy.get("interview_style", random.choice(["Formal", "Casual"])),
                "interview_attitude": strategy.get("interview_attitude", random.choice(["Humble", "Confident"])),
                "skills_to_improve": strategy.get("skills_to_improve", []),
                "lyingness": strategy.get("lyingness", random.uniform(0.0, 0.3))
            })
            
            candidate = CandidateAgent(i, profile, llm_config)
            candidates.append(candidate)
    else:
        # Generate random candidates
        for i in range(num_candidates):
            profile = generate_candidate_profile(i, llm_config)
            # Add random strategy parameters
            profile.update({
                "interview_style": random.choice(["Formal", "Casual"]),
                "interview_attitude": random.choice(["Humble", "Confident"]),
                "skills_to_improve": random.sample(
                    list(profile.get("skills", {}).keys()),
                    min(2, len(profile.get("skills", {})))
                ),
                "lyingness": random.uniform(0.0, 0.3)
            })
            candidate = CandidateAgent(i, profile, llm_config)
            candidates.append(candidate)
    
    # Generate company profiles
    print(f"Generating {num_companies} company profiles...")
    companies = []
    for i in range(num_companies):
        profile = generate_company_profile(i, llm_config)
        # Set appropriate budget for French junior positions
        profile["budget"] = random.randint(35000, 42000)  # EUR for junior positions
        company = CompanyAgent(i, profile, llm_config)
        companies.append(company)
    
    # Track simulation statistics
    stats = {
        "rounds": [],
        "overall": {
            "candidates_hired": 0,
            "total_applications": 0,
            "total_interviews": 0,
            "total_offers": 0,
            "rejected_offers": 0,
            "avg_candidate_reward": 0.0,
            "avg_company_reward": 0.0
        },
        "strategy_changes": {
            "candidates": {c.id: [] for c in candidates},
            "companies": {c.id: [] for c in companies}
        }
    }
    
    # Run simulation rounds
    for round_num in range(num_rounds):
        print(f"\n===== SIMULATION ROUND {round_num + 1}/{num_rounds} =====")
        
        # If not the first round, let agents observe and learn from others' strategies
        if round_num > 0:
            print("\nAgents analyzing previous round results...")
            
            # Find candidates who were caught lying
            caught_lying_candidates = []
            for candidate in candidates:
                # Check if candidate was caught lying in previous round
                if candidate.reward_history and candidate.reward_history[-1] < 0.1:
                    for record in stats["rounds"][-1].get("interviews", []):
                        if record.get("candidate_id") == candidate.id and record.get("lying_detected", False):
                            caught_lying_candidates.append(candidate.id)
                            break
            
            # Let other candidates learn from those caught lying
            if caught_lying_candidates:
                print(f"  Candidates observing others who were caught lying: {caught_lying_candidates}")
                for candidate in candidates:
                    if candidate.id not in caught_lying_candidates and candidate.risk_tolerance < 0.7:
                        # Low-risk candidates learn from others' mistakes
                        if candidate.lyingness > 0.1:
                            old_lyingness = candidate.lyingness
                            candidate.lyingness = max(0.0, candidate.lyingness - 0.05)
                            print(f"  - Candidate {candidate.id} reduced lyingness from {old_lyingness:.2f} to {candidate.lyingness:.2f} after observing others")
            
            # Find companies that failed to hire anyone
            failed_companies = []
            for company in companies:
                if company.reward_history and company.reward_history[-1] < 0.1:
                    failed_companies.append(company.id)
            
            # Let companies learn from successful ones
            if failed_companies and len(failed_companies) < len(companies):
                print(f"  Companies observing those that failed to hire: {failed_companies}")
                successful_companies = [c for c in companies if c.id not in failed_companies]
                
                if successful_companies:
                    # Calculate average standards of successful companies
                    avg_standards = sum(c.hiring_standards for c in successful_companies) / len(successful_companies)
                    
                    for company in companies:
                        if company.id in failed_companies:
                            # Failed companies adjust toward successful companies' standards
                            if company.hiring_standards > avg_standards + 0.1:
                                old_standards = company.hiring_standards
                                company.hiring_standards = max(0.4, company.hiring_standards - 0.08)
                                print(f"  - Company {company.id} reduced standards from {old_standards:.2f} to {company.hiring_standards:.2f} based on market")
        
        # Run simulation round
        round_stats = run_simulation_round(
            candidates=candidates,
            companies=companies,
            llm_config=llm_config,
            round_num=round_num
        )
        
        # Update statistics
        stats["rounds"].append(round_stats)
        
        # Update overall stats
        stats["overall"]["candidates_hired"] += round_stats["candidates_hired"]
        stats["overall"]["total_applications"] += round_stats["total_applications"]
        stats["overall"]["total_interviews"] += round_stats["total_interviews"]
        stats["overall"]["total_offers"] += round_stats["total_offers"]
        stats["overall"]["rejected_offers"] += round_stats["rejected_offers"]
        
        # Enhanced tracking - Round data
        round_data = {
            "id": round_num,
            "timestamp": time.time(),
            "hired_count": round_stats["candidates_hired"],
            "application_count": round_stats["total_applications"],
            "interview_count": round_stats["total_interviews"],
            "offer_count": round_stats["total_offers"],
            "rejected_offer_count": round_stats["rejected_offers"],
            "avg_candidate_reward": round_stats["avg_candidate_reward"],
            "avg_company_reward": round_stats["avg_company_reward"],
            "duration": round_stats["duration"]
        }
        temporal_data["rounds"].append(round_data)
        
        # Enhanced tracking - Candidate data 
        for candidate in candidates:
            cid = candidate.id
            
            # Skill tracking
            skill_snapshot = {
                "round": round_num,
                "skills": {k: v for k, v in candidate.profile.get("skills", {}).items()},
                "initial_skills": {k: v for k, v in candidate.profile.get("initial_skills", {}).items()}
            }
            temporal_data["candidates"][cid]["skills"].append(skill_snapshot)
            
            # Strategy tracking
            strategy_snapshot = {
                "round": round_num,
                "lyingness": candidate.lyingness,
                "truth_level": candidate.truth_level,
                "interview_style": candidate.interview_style,
                "interview_attitude": candidate.interview_attitude,
                "risk_tolerance": candidate.risk_tolerance,
                "application_strategy": {
                    "targeted_applications": candidate.application_strategy.get("targeted_applications", 0.75),
                    "skill_emphasis": candidate.application_strategy.get("skill_emphasis", [])
                }
            }
            temporal_data["candidates"][cid]["strategies"].append(strategy_snapshot)
            
            # Reward tracking
            reward = round_stats["candidate_rewards"].get(cid, 0.0)
            temporal_data["candidates"][cid]["rewards"].append({
                "round": round_num,
                "reward": reward,
                "cumulative_reward": sum(candidate.reward_history) if hasattr(candidate, "reward_history") else 0
            })
            
            # Interview and offer tracking
            interviews_this_round = []
            for interview in round_stats.get("interviews", []):
                if interview.get("candidate_id") == cid:
                    interviews_this_round.append({
                        "company_id": interview.get("company_id"),
                        "score": next((r.get("score", 0) for r in companies[interview.get("company_id")].interview_history 
                                      if r.get("interview_id") == interview.get("interview_id")), 0),
                        "lying_detected": next((r.get("lying_detected", False) for r in companies[interview.get("company_id")].interview_history 
                                             if r.get("interview_id") == interview.get("interview_id")), False)
                    })
                    
                    # Add to network data
                    network_data["interviews"].append({
                        "round": round_num,
                        "candidate_id": cid,
                        "company_id": interview.get("company_id"),
                        "job_id": interview.get("job_id"),
                        "lying_detected": next((r.get("lying_detected", False) for r in companies[interview.get("company_id")].interview_history 
                                             if r.get("interview_id") == interview.get("interview_id")), False)
                    })
            
            temporal_data["candidates"][cid]["interviews"].append({
                "round": round_num,
                "count": len(interviews_this_round),
                "details": interviews_this_round
            })
            
            # Offer tracking
            offers_this_round = []
            for offer in round_stats.get("offers", []):
                if offer.get("candidate_id") == cid:
                    offers_this_round.append({
                        "company_id": offer.get("company_id"),
                        "job_id": offer.get("job_id"),
                        "salary": offer.get("salary"),
                        "accepted": next((d.get("accepted", False) for d in round_stats.get("candidate_decisions", [])
                                       if d.get("candidate_id") == cid and d.get("company_id") == offer.get("company_id")), False)
                    })
                    
                    # Add to network data
                    network_data["offers"].append({
                        "round": round_num,
                        "candidate_id": cid,
                        "company_id": offer.get("company_id"),
                        "job_id": offer.get("job_id"),
                        "salary": offer.get("salary"),
                        "accepted": next((d.get("accepted", False) for d in round_stats.get("candidate_decisions", [])
                                       if d.get("candidate_id") == cid and d.get("company_id") == offer.get("company_id")), False)
                    })
            
            temporal_data["candidates"][cid]["offers"].append({
                "round": round_num,
                "count": len(offers_this_round),
                "details": offers_this_round
            })
            
            # Correlation data
            correlation_data["lyingness_vs_success"].append({
                "round": round_num,
                "candidate_id": cid,
                "lyingness": candidate.lyingness,
                "reward": reward,
                "interviews": len(interviews_this_round),
                "offers": len(offers_this_round),
                "accepted_offer": any(o.get("accepted", False) for o in offers_this_round),
                "truth_level": candidate.truth_level,
                "risk_tolerance": candidate.risk_tolerance
            })
            
            # Skills vs interviews correlation
            skills_data = {
                "round": round_num,
                "candidate_id": cid,
                "avg_skill_level": sum(candidate.profile.get("skills", {}).values()) / max(1, len(candidate.profile.get("skills", {}))),
                "max_skill_level": max(candidate.profile.get("skills", {}).values()) if candidate.profile.get("skills", {}) else 0,
                "interview_count": len(interviews_this_round),
                "offer_count": len(offers_this_round)
            }
            correlation_data["skills_vs_interviews"].append(skills_data)
        
        # Enhanced tracking - Company data
        for company in companies:
            cid = company.id
            
            # Strategy tracking
            standards_snapshot = {
                "round": round_num,
                "hiring_standards": company.hiring_standards,
                "budget_flexibility": company.budget_flexibility,
                "skill_emphasis": company.skill_emphasis.copy() if hasattr(company, "skill_emphasis") else []
            }
            temporal_data["companies"][cid]["standards"].append(standards_snapshot)
            
            # Reward tracking
            reward = round_stats["company_rewards"].get(cid, 0.0)
            temporal_data["companies"][cid]["rewards"].append({
                "round": round_num,
                "reward": reward,
                "cumulative_reward": sum(company.reward_history) if hasattr(company, "reward_history") else 0
            })
            
            # Applicant tracking
            applicants_count = sum(1 for app in round_stats.get("applications", []) if app.get("company_id") == cid)
            shortlisted_count = sum(1 for s in round_stats.get("shortlists", {}).get(cid, []) if s.get("company_id") == cid)
            interviewed_count = sum(1 for i in round_stats.get("interviews", []) if i.get("company_id") == cid)
            offers_count = sum(1 for o in round_stats.get("offers", []) if o.get("company_id") == cid)
            hired_count = sum(1 for h in round_stats.get("hiring_results", []) 
                            if h.get("company_id") == cid and h.get("accepted", False))
            
            applicants_snapshot = {
                "round": round_num,
                "total": applicants_count,
                "shortlisted": shortlisted_count,
                "interviewed": interviewed_count,
                "offered": offers_count,
                "hired": hired_count
            }
            temporal_data["companies"][cid]["applicants"].append(applicants_snapshot)
            
            # Hire data
            hired_candidates = []
            for hire in round_stats.get("hiring_results", []):
                if hire.get("company_id") == cid and hire.get("accepted", False):
                    candidate_id = hire.get("candidate_id")
                    job_id = hire.get("job_id")
                    
                    # Find the offer with salary
                    salary = next((o.get("salary", 0) for o in round_stats.get("offers", [])
                                 if o.get("company_id") == cid and o.get("candidate_id") == candidate_id), 0)
                    
                    hired_candidates.append({
                        "candidate_id": candidate_id,
                        "job_id": job_id,
                        "salary": salary
                    })
                    
                    # Add to network data
                    network_data["hires"].append({
                        "round": round_num,
                        "candidate_id": candidate_id,
                        "company_id": cid,
                        "job_id": job_id,
                        "salary": salary
                    })
                    
                    # Budget vs salary correlation
                    budget = company.profile.get("budget", 38000)
                    correlation_data["budget_vs_salary"].append({
                        "round": round_num,
                        "company_id": cid,
                        "budget": budget,
                        "salary": salary,
                        "ratio": salary / budget if budget > 0 else 0,
                        "budget_flexibility": company.budget_flexibility,
                        "candidate_id": candidate_id
                    })
            
            temporal_data["companies"][cid]["hires"].append({
                "round": round_num,
                "count": len(hired_candidates),
                "details": hired_candidates
            })
            
            # Correlation data
            correlation_data["standards_vs_success"].append({
                "round": round_num,
                "company_id": cid,
                "hiring_standards": company.hiring_standards,
                "budget_flexibility": company.budget_flexibility,
                "applicants": applicants_count,
                "shortlisted": shortlisted_count,
                "interviewed": interviewed_count,
                "offered": offers_count,
                "hired": hired_count,
                "reward": reward,
                "sector": company.profile.get("sector", "Unknown"),
                "budget": company.profile.get("budget", 38000)
            })
        
        # Calculate average rewards per round
        if round_num == 0:
            stats["overall"]["avg_candidate_reward"] = round_stats["avg_candidate_reward"]
            stats["overall"]["avg_company_reward"] = round_stats["avg_company_reward"]
        else:
            # Weighted average
            prev_avg_candidate = stats["overall"]["avg_candidate_reward"]
            prev_avg_company = stats["overall"]["avg_company_reward"]
            
            stats["overall"]["avg_candidate_reward"] = (prev_avg_candidate * round_num + round_stats["avg_candidate_reward"]) / (round_num + 1)
            stats["overall"]["avg_company_reward"] = (prev_avg_company * round_num + round_stats["avg_company_reward"]) / (round_num + 1)
    
    # After all rounds, calculate final statistics
    final_stats = {
        "simulation_id": session_id,
        "time": {
            "start": start_time,
            "end": time.time(),
            "duration": time.time() - start_time,
            "avg_round_time": sum(r.get("duration", 0) for r in stats["rounds"]) / num_rounds
        },
        "outcomes": {
            "total_candidates": num_candidates,
            "total_companies": num_companies,
            "total_hires": stats["overall"]["candidates_hired"],
            "total_applications": stats["overall"]["total_applications"],
            "total_interviews": stats["overall"]["total_interviews"],
            "total_offers": stats["overall"]["total_offers"],
            "avg_applications_per_candidate": stats["overall"]["total_applications"] / num_candidates,
            "avg_interviews_per_company": stats["overall"]["total_interviews"] / num_companies,
            "hire_rate": stats["overall"]["candidates_hired"] / (num_candidates * num_rounds),
            "avg_time_to_hire": sum(r.get("duration", 0) for r in stats["rounds"]) / max(1, stats["overall"]["candidates_hired"])
        },
        "candidate_evolution": {
            "skill_improvements": {},
            "strategy_changes": {},
            "reward_progression": {}
        },
        "company_evolution": {
            "standard_changes": {},
            "budget_flexibility_changes": {},
            "reward_progression": {}
        }
    }
    
    # Calculate skill and strategy evolution metrics
    for candidate in candidates:
        cid = candidate.id
        
        # Skill evolution
        initial_skills = candidate.profile.get("initial_skills", {})
        final_skills = candidate.profile.get("skills", {})
        
        skill_improvements = {}
        for skill, final_level in final_skills.items():
            initial_level = initial_skills.get(skill, 0)
            if final_level > initial_level:
                skill_improvements[skill] = {
                    "initial": initial_level, 
                    "final": final_level,
                    "improvement": final_level - initial_level
                }
        
        final_stats["candidate_evolution"]["skill_improvements"][str(cid)] = skill_improvements
        
        # Strategy evolution
        strategy_changes = {}
        if hasattr(candidate, "truth_level"):
            initial_truth = candidate.profile.get("truth_level", 1.0)
            strategy_changes["truth_level"] = {
                "initial": initial_truth,
                "final": candidate.truth_level,
                "change": candidate.truth_level - initial_truth
            }
        
        if hasattr(candidate, "lyingness"):
            initial_lyingness = candidate.profile.get("lyingness", 0.0)
            strategy_changes["lyingness"] = {
                "initial": initial_lyingness,
                "final": candidate.lyingness,
                "change": candidate.lyingness - initial_lyingness
            }
        
        final_stats["candidate_evolution"]["strategy_changes"][str(cid)] = strategy_changes
        
        # Reward progression
        if hasattr(candidate, "reward_history") and candidate.reward_history:
            final_stats["candidate_evolution"]["reward_progression"][str(cid)] = {
                "total": sum(candidate.reward_history),
                "avg": sum(candidate.reward_history) / len(candidate.reward_history),
                "final": candidate.reward_history[-1],
                "trend": "increasing" if len(candidate.reward_history) > 1 and 
                        candidate.reward_history[-1] > candidate.reward_history[0] else "decreasing"
            }
    
    # Company evolution metrics
    for company in companies:
        cid = company.id
        
        # Standards evolution
        initial_standards = company.profile.get("hiring_standards", 0.7)
        standard_changes = {
            "initial": initial_standards,
            "final": company.hiring_standards,
            "change": company.hiring_standards - initial_standards
        }
        final_stats["company_evolution"]["standard_changes"][str(cid)] = standard_changes
        
        # Budget flexibility evolution
        initial_flex = company.profile.get("budget_flexibility", 0.2)
        flex_changes = {
            "initial": initial_flex,
            "final": company.budget_flexibility,
            "change": company.budget_flexibility - initial_flex
        }
        final_stats["company_evolution"]["budget_flexibility_changes"][str(cid)] = flex_changes
        
        # Reward progression
        if hasattr(company, "reward_history") and company.reward_history:
            final_stats["company_evolution"]["reward_progression"][str(cid)] = {
                "total": sum(company.reward_history),
                "avg": sum(company.reward_history) / len(company.reward_history),
                "final": company.reward_history[-1],
                "trend": "increasing" if len(company.reward_history) > 1 and 
                        company.reward_history[-1] > company.reward_history[0] else "decreasing"
            }
    
    # Combine all data for comprehensive results
    results = {
        "simulation_id": session_id,
        "config": {
            "num_candidates": num_candidates,
            "num_companies": num_companies,
            "num_rounds": num_rounds,
            "candidate_type": "Junior from French universities"
        },
        "stats": stats,
        "temporal_data": temporal_data,
        "correlation_data": correlation_data,
        "network_data": network_data,
        "final_stats": final_stats,
        "candidates": [
            {
                "id": c.id,
                "profile": c.profile,
                "final_state": {
                    "lyingness": c.lyingness if hasattr(c, "lyingness") else 0.0,
                    "truth_level": c.truth_level if hasattr(c, "truth_level") else 1.0,
                    "interview_style": c.interview_style if hasattr(c, "interview_style") else "Unknown",
                    "interview_attitude": c.interview_attitude if hasattr(c, "interview_attitude") else "Unknown",
                    "skills": c.profile.get("skills", {}),
                    "total_rewards": sum(c.reward_history) if hasattr(c, "reward_history") else 0
                }
            } for c in candidates
        ],
        "companies": [
            {
                "id": c.id,
                "profile": c.profile,
                "final_state": {
                    "hiring_standards": c.hiring_standards if hasattr(c, "hiring_standards") else 0.7,
                    "budget_flexibility": c.budget_flexibility if hasattr(c, "budget_flexibility") else 0.2,
                    "skill_emphasis": c.skill_emphasis if hasattr(c, "skill_emphasis") else [],
                    "total_rewards": sum(c.reward_history) if hasattr(c, "reward_history") else 0
                }
            } for c in companies
        ]
    }
    
    # Save to file
    filename = f"{output_dir}/simulation_{session_id}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to {filename}")
    
    # Print summary
    print("\n===== SIMULATION SUMMARY =====")
    print(f"Total candidates hired: {stats['overall']['candidates_hired']}/{num_candidates * num_rounds}")
    print(f"Total applications: {stats['overall']['total_applications']}")
    print(f"Total interviews: {stats['overall']['total_interviews']}")
    print(f"Total offers: {stats['overall']['total_offers']}")
    print(f"Rejected offers: {stats['overall']['rejected_offers']}")
    print(f"Average candidate reward: {stats['overall']['avg_candidate_reward']:.2f}")
    print(f"Average company reward: {stats['overall']['avg_company_reward']:.2f}")

    print("\n===== CANDIDATE STRATEGY EVOLUTION =====")
    for candidate in candidates:
        initial_profile = getattr(candidate, "initial_profile", candidate.profile)
        
        print(f"Candidate {candidate.id}:")
        print(f"  Truth level: {initial_profile.get('truth_level', 1.0):.2f} → {candidate.truth_level:.2f}")
        print(f"  Lyingness: {initial_profile.get('lyingness', 0.0):.2f} → {candidate.lyingness:.2f}")
        print(f"  Interview style: {initial_profile.get('interview_style', 'Unknown')} → {candidate.interview_style}")
        print(f"  Interview attitude: {initial_profile.get('interview_attitude', 'Unknown')} → {candidate.interview_attitude}")
        
        # Show skill improvements
        improved_skills = []
        for skill, level in candidate.profile.get("skills", {}).items():
            initial_level = initial_profile.get("skills", {}).get(skill, 0)
            if level > initial_level:
                improved_skills.append(f"{skill}: {initial_level} → {level}")
        
        if improved_skills:
            print(f"  Skills improved:")
            for skill_improvement in improved_skills:
                print(f"    * {skill_improvement}")
        print()

    print("\n===== COMPANY STRATEGY EVOLUTION =====")
    for company in companies:
        initial_profile = company.profile
        
        print(f"Company {company.id} ({company.profile.get('name', 'Unknown')}):")
        print(f"  Hiring standards: {initial_profile.get('hiring_standards', 0.7):.2f} → {company.hiring_standards:.2f}")
        print(f"  Budget flexibility: {initial_profile.get('budget_flexibility', 0.2):.2f} → {company.budget_flexibility:.2f}")
        
        # Track if standards were significantly lowered
        if company.hiring_standards < initial_profile.get('hiring_standards', 0.7) * 0.8:
            print(f"  * Significantly lowered hiring standards due to candidate shortage or rejections")
        
        if company.budget_flexibility > initial_profile.get('budget_flexibility', 0.2) * 1.5:
            print(f"  * Significantly increased budget flexibility to attract candidates")
        print()
    
    # Return results
    return results