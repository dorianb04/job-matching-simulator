"""
Agent definitions for the simulation.
"""
import autogen
from typing import Dict, Any, List, Optional
from .monitoring import AgentOpsTracker

def get_llm_config(api_key: str, model: str = "llama3-70b-8192") -> Dict[str, Any]:
    """
    Get LLM configuration for Autogen.
    
    Args:
        api_key: Groq API key
        model: Model name
        
    Returns:
        Dictionary with LLM configuration
    """
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": "https://api.groq.com/openai/v1",
            }
        ],
        "temperature": 0.7,
    }

def create_user_proxy(name: str = "user_proxy") -> autogen.UserProxyAgent:
    """
    Create a user proxy agent with Docker disabled.
    
    Args:
        name: Name for the agent
        
    Returns:
        UserProxyAgent instance
    """
    from .config import CODE_EXECUTION_CONFIG
    
    return autogen.UserProxyAgent(
        name=name,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=CODE_EXECUTION_CONFIG
    )

class CandidateAgent:
    """Agent representing a job candidate."""
    
    def __init__(
        self, 
        candidate_id: int, 
        profile: Dict[str, Any], 
        llm_config: Dict[str, Any],
        tracker: Optional[AgentOpsTracker] = None
    ):
        """
        Initialize a candidate agent.
        
        Args:
            candidate_id: Unique identifier
            profile: Candidate profile data
            llm_config: LLM configuration
            tracker: AgentOps tracker
        """
        self.id = candidate_id
        self.profile = profile
        self.tracker = tracker
        
        # Create the Autogen agent
        self.agent = autogen.AssistantAgent(
            name=f"candidate_{candidate_id}",
            llm_config=llm_config,
            system_message=self._create_system_message()
        )
        
        # Track agent creation
        if tracker:
            tracker.track_event("agent_created", {
                "agent_id": f"candidate_{candidate_id}",
                "agent_type": "candidate",
                "profile": profile
            })
    
    def _create_system_message(self) -> str:
        """Create the system message for the agent."""
        return f"""
        You are a job candidate with ID {self.id}.
        
        Your profile:
        - Energy: {self.profile['energy']}
        - Motivation: {self.profile['motivation']}
        - Money importance: {self.profile['money_importance']}
        - Skills: {self.profile['skills']}
        
        You're looking for a job that matches your skills and preferences.
        """
    
    def evaluate_job_offer(self, job_offer: Dict[str, Any]) -> float:
        """
        Evaluate a job offer based on preferences.
        
        Args:
            job_offer: Job offer details
            
        Returns:
            Score between 0 and 1
        """
        # Simple scoring logic (to be expanded)
        score = 0.0
        
        # Check for matching skills
        required_skills = job_offer.get("required_skills", [])
        candidate_skills = self.profile.get("skills", {})
        
        skill_match = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 0
        
        # Consider sector motivation
        sector = job_offer.get("sector", "")
        sector_motivation = self.profile.get("motivation", {}).get(sector, 0)
        
        # Consider salary
        salary = job_offer.get("salary", 0)
        money_importance = self.profile.get("money_importance", 0.5)
        
        # Combine factors
        score = (
            0.4 * skill_match +
            0.3 * (sector_motivation / 10) +
            0.3 * (min(salary, 100000) / 100000) * money_importance
        )
        
        # Track evaluation
        if self.tracker:
            self.tracker.track_event("job_offer_evaluated", {
                "candidate_id": self.id,
                "job_id": job_offer.get("id"),
                "score": score
            })
        
        return score

class CompanyAgent:
    """Agent representing a company."""
    
    def __init__(
        self, 
        company_id: int, 
        profile: Dict[str, Any], 
        llm_config: Dict[str, Any],
        tracker: Optional[AgentOpsTracker] = None
    ):
        """
        Initialize a company agent.
        
        Args:
            company_id: Unique identifier
            profile: Company profile data
            llm_config: LLM configuration
            tracker: AgentOps tracker
        """
        self.id = company_id
        self.profile = profile
        self.tracker = tracker
        
        # Create the Autogen agent
        self.agent = autogen.AssistantAgent(
            name=f"company_{company_id}",
            llm_config=llm_config,
            system_message=self._create_system_message()
        )
        
        # Track agent creation
        if tracker:
            tracker.track_event("agent_created", {
                "agent_id": f"company_{company_id}",
                "agent_type": "company",
                "profile": profile
            })
    
    def _create_system_message(self) -> str:
        """Create the system message for the agent."""
        return f"""
        You are a company with ID {self.id}.
        
        Your profile:
        - Name: {self.profile['name']}
        - Sector: {self.profile['sector']}
        - Budget: {self.profile['budget']}
        - Required skills: {self.profile['required_skills']}
        
        You're looking to hire candidates that match your requirements.
        """
    
    def evaluate_candidate(self, candidate_profile: Dict[str, Any]) -> float:
        """
        Evaluate a candidate based on company requirements.
        
        Args:
            candidate_profile: Candidate profile
            
        Returns:
            Score between 0 and 1
        """
        # Simple scoring logic (to be expanded)
        score = 0.0
        
        # Check for matching skills
        required_skills = self.profile.get("required_skills", [])
        candidate_skills = candidate_profile.get("skills", {})
        
        skill_match = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 0
        
        # Budget considerations
        budget = self.profile.get("budget", 0)
        expected_salary = candidate_profile.get("expected_salary", budget)
        budget_fit = 1.0 if expected_salary <= budget else budget / expected_salary
        
        # Combine factors
        score = 0.7 * skill_match + 0.3 * budget_fit
        
        # Track evaluation
        if self.tracker:
            self.tracker.track_event("candidate_evaluated", {
                "company_id": self.id,
                "candidate_id": candidate_profile.get("id"),
                "score": score
            })
        
        return score

def create_interviewer_agent(
    company_name: str,
    position: str,
    style: str,
    llm_config: Dict[str, Any],
    tracker: Optional[AgentOpsTracker] = None
) -> autogen.AssistantAgent:
    """
    Create an interviewer agent.
    
    Args:
        company_name: Name of the company
        position: Job position
        style: Interview style
        llm_config: LLM configuration
        tracker: AgentOps tracker
        
    Returns:
        Autogen assistant agent
    """
    # Create a name without spaces
    # Replace spaces with underscores and remove special characters
    safe_company_name = company_name.replace(" ", "_").replace("-", "").replace("/", "")
    safe_position = position.replace(" ", "_").replace("-", "").replace("/", "")
    
    agent_name = f"interviewer_{safe_company_name}_{safe_position}"
    
    system_message = f"""
    You are an interviewer at {company_name}.
    You are conducting interviews for the {position} position.
    Your interview style is {style}.
    
    Ask relevant questions to assess the candidate's skills and fit for the position.
    """
    
    interviewer = autogen.AssistantAgent(
        name=agent_name,
        llm_config=llm_config,
        system_message=system_message
    )
    
    # Track agent creation
    if tracker:
        tracker.track_event("agent_created", {
            "agent_id": agent_name,
            "agent_type": "interviewer",
            "company": company_name,
            "position": position,
            "style": style
        })
    
    return interviewer