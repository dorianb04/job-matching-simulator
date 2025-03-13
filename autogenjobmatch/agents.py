"""
Agent definitions for the job matching simulation.
"""
import autogen
import random
from typing import Dict, Any, List, Optional

from .config import (
    GROQ_MODEL,
    DATA_SCIENCE_SKILLS,
    SECTORS,
    INITIAL_ENERGY,
    INTERVIEW_ATTITUDES,
    INTERVIEW_STRATEGIES,
    CODE_EXECUTION_CONFIG,
    EVALUATION_WEIGHTS,
    CANDIDATE_EVALUATION_WEIGHTS,
    CANDIDATE_PROMPT,
    COMPANY_PROMPT
)

def get_llm_config(api_key: str, model: str = GROQ_MODEL) -> Dict[str, Any]:
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
                "price" : [0, 0]
            },
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
        llm_config: Dict[str, Any]
    ):
        """
        Initialize a candidate agent.
        
        Args:
            candidate_id: Unique identifier
            profile: Candidate profile data
            llm_config: LLM configuration
        """
        self.id = candidate_id
        self.profile = profile
        
        # Create safe agent name (no spaces)
        safe_name = f"candidate_{candidate_id}"
        
        # Create the Autogen agent
        self.agent = autogen.AssistantAgent(
            name=safe_name,
            llm_config=llm_config,
            system_message=self._create_system_message()
        )
    
    def _create_system_message(self) -> str:
        """Create the system message for the agent."""
        return CANDIDATE_PROMPT.format(
            energy=self.profile.get("energy", INITIAL_ENERGY),
            motivation=", ".join([f"{k}: {v}" for k, v in self.profile.get("motivation", {}).items()]),
            money_importance=self.profile.get("money_importance", 0.5),
            skills=", ".join([f"{k}: {v}" for k, v in self.profile.get("skills", {}).items()])
        )
    
    def evaluate_job_offer(self, job_offer: Dict[str, Any]) -> float:
        """
        Evaluate a job offer based on preferences.
        
        Args:
            job_offer: Job offer details
            
        Returns:
            Score between 0 and 1
        """
        # Get weights
        weights = EVALUATION_WEIGHTS
        
        # Check for matching skills
        required_skills = job_offer.get("required_skills", [])
        candidate_skills = self.profile.get("skills", {})
        
        skill_match = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 0
        
        # Consider sector motivation
        sector = job_offer.get("sector", "")
        sector_motivation = self.profile.get("motivation", {}).get(sector, 0) / 10  # Normalize to 0-1
        
        # Consider salary
        salary_range = job_offer.get("salary_range", (0, 0))
        avg_salary = sum(salary_range) / 2 if salary_range else 0
        salary_score = min(avg_salary, 150000) / 150000  # Normalize to 0-1
        money_importance = self.profile.get("money_importance", 0.5)
        
        # Combine factors using weights
        score = (
            weights["skill_match"] * skill_match +
            weights["sector_motivation"] * sector_motivation +
            weights["salary_match"] * salary_score * money_importance
        )
        
        return score
    
    def create_cv(self, truth_level: float = 1.0) -> Dict[str, Any]:
        """
        Create a CV, potentially with exaggerations.
        
        Args:
            truth_level: How truthful to be (1.0 = completely honest)
            
        Returns:
            CV data
        """
        skills = {}
        
        # Copy skills, potentially with exaggerations
        for skill, level in self.profile.get("skills", {}).items():
            if random.random() > truth_level:
                # Exaggerate skill level (but not beyond 10)
                exaggerated_level = min(level + random.randint(1, 3), 10)
                skills[skill] = exaggerated_level
            else:
                skills[skill] = level
        
        return {
            "candidate_id": self.id,
            "skills": skills,
            "truth_level": truth_level,
            "expected_salary": random.randint(50000, 150000)
        }
    
    def attend_interview(self, strategy: str = None, attitude: str = None) -> Dict[str, Any]:
        """
        Prepare for interview with a strategy.
        
        Args:
            strategy: Interview strategy (Formal/Casual)
            attitude: Interview attitude (Humble/Confident)
            
        Returns:
            Interview preparation data
        """
        # Default strategy and attitude if not specified
        if strategy is None:
            strategy = random.choice(INTERVIEW_STRATEGIES)
        
        if attitude is None:
            attitude = random.choice(INTERVIEW_ATTITUDES)
        
        # Consume energy
        self.profile["energy"] = max(0, self.profile.get("energy", 0) - 1)
        
        return {
            "candidate_id": self.id,
            "strategy": strategy,
            "attitude": attitude,
            "energy": self.profile["energy"]
        }

class CompanyAgent:
    """Agent representing a company."""
    
    def __init__(
        self, 
        company_id: int, 
        profile: Dict[str, Any], 
        llm_config: Dict[str, Any]
    ):
        """
        Initialize a company agent.
        
        Args:
            company_id: Unique identifier
            profile: Company profile data
            llm_config: LLM configuration
        """
        self.id = company_id
        self.profile = profile
        
        # Create safe agent name (no spaces)
        safe_name = f"company_{company_id}"
        
        # Create the Autogen agent
        self.agent = autogen.AssistantAgent(
            name=safe_name,
            llm_config=llm_config,
            system_message=self._create_system_message()
        )
    
    def _create_system_message(self) -> str:
        """Create the system message for the agent."""
        return COMPANY_PROMPT.format(
            sector=self.profile.get("sector", "Technology"),
            name=self.profile.get("name", f"Company {self.id}"),
            budget=self.profile.get("budget", 100000),
            skills=", ".join(self.profile.get("required_skills", []))
        )
    
    def evaluate_candidate(self, candidate_profile: Dict[str, Any]) -> float:
        """
        Evaluate a candidate based on company requirements.
        
        Args:
            candidate_profile: Candidate profile
            
        Returns:
            Score between 0 and 1
        """
        # Get weights
        weights = CANDIDATE_EVALUATION_WEIGHTS
        
        # Check for matching skills
        required_skills = self.profile.get("required_skills", [])
        candidate_skills = candidate_profile.get("skills", {})
        
        skill_match = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 0
        
        # Budget considerations
        budget = self.profile.get("budget", 0)
        expected_salary = candidate_profile.get("expected_salary", budget)
        budget_fit = 1.0 if expected_salary <= budget else budget / expected_salary
        
        # Combine factors using weights
        score = (
            weights["skill_match"] * skill_match +
            weights["budget_fit"] * budget_fit
        )
        
        return score
    
    def select_candidates(self, applications: List[Dict[str, Any]], num_slots: int = 2) -> List[int]:
        """
        Select candidates for interviews.
        
        Args:
            applications: List of applications
            num_slots: Number of interview slots
            
        Returns:
            List of selected candidate IDs
        """
        # Filter applications for this company
        company_applications = [
            app for app in applications if app.get("company_id") == self.id
        ]
        
        # If fewer applications than slots, select all
        if len(company_applications) <= num_slots:
            return [app.get("candidate_id") for app in company_applications]
        
        # Otherwise, select top scoring applications
        company_applications.sort(key=lambda x: x.get("score", 0), reverse=True)
        return [app.get("candidate_id") for app in company_applications[:num_slots]]
    
    def make_hiring_decision(self, interview_result: Dict[str, Any]) -> bool:
        """
        Make a hiring decision based on an interview.
        
        Args:
            interview_result: Results from the interview
            
        Returns:
            Whether to hire the candidate
        """
        # This is a simplified implementation
        # In a real system, you would analyze the interview responses
        
        # For now, just make a random decision weighted by the number of messages
        # This simulates longer, more engaged interviews having better chances
        num_messages = len(interview_result.get("messages", []))
        base_chance = 0.5
        
        # Adjust chance based on number of messages (more messages = better chance)
        adjusted_chance = min(0.8, base_chance + (num_messages / 20))
        
        return random.random() < adjusted_chance

def create_interviewer_agent(
    company_name: str,
    position: str,
    style: str,
    llm_config: Dict[str, Any]
) -> autogen.AssistantAgent:
    """
    Create an interviewer agent.
    
    Args:
        company_name: Name of the company
        position: Job position
        style: Interview style
        llm_config: LLM configuration
        
    Returns:
        Autogen assistant agent
    """
    # Create a name without spaces
    # Replace spaces with underscores and remove special characters
    safe_company_name = company_name.replace(" ", "_").replace("-", "").replace("/", "")
    safe_position = position.replace(" ", "_").replace("-", "").replace("/", "")
    
    agent_name = f"interviewer_{safe_company_name}_{safe_position}"[:50]  # Limit length
    
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
    
    return interviewer