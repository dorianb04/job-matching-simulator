"""
Agent definitions for the job matching simulation.
"""
import autogen
import random
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

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
    COMPANY_PROMPT,
    MIN_SALARY,
    MAX_SALARY,
    LIE_PROBABILITY,
    STRATEGIC_LEARNING_RATE,
    MAX_APPLICATIONS_PER_CANDIDATE
)

def get_llm_config(api_key: str, model: str = GROQ_MODEL, verbose: bool = False) -> Dict[str, Any]:
    """
    Get LLM configuration for Autogen.
    
    Args:
        api_key: Groq API key
        model: Model name
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary with LLM configuration
    """
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": "https://api.groq.com/openai/v1",
                "price": [0, 0]
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
    """Agent representing a job candidate with strategic capabilities."""
    
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
        self.llm_config = llm_config
        
        # Strategic parameters
        self.truth_level = random.uniform(0.7, 1.0)  # Honesty level in applications
        self.risk_tolerance = random.uniform(0.3, 0.8)  # Willingness to take risks
        
        # Interview strategy parameters
        self.interview_style = profile.get("interview_style", random.choice(["Formal", "Casual"]))
        self.interview_attitude = profile.get("interview_attitude", random.choice(["Humble", "Confident"]))
        self.lyingness = profile.get("lyingness", random.uniform(0.0, 0.3))  # How much the candidate lies
        
        # Skills to improve after simulation
        self.skills_to_improve = profile.get("skills_to_improve", random.sample(
            list(self.profile.get("skills", {}).keys()),
            min(2, len(self.profile.get("skills", {})))
        ))
        
        self.application_strategy = {
            "targeted_applications": random.uniform(0.5, 1.0),  # Focus on fitting jobs vs. quantity
            "skill_emphasis": random.sample(
                list(self.profile.get("skills", {}).keys()),  # Convert to list
                min(3, len(self.profile.get("skills", {})))
            )  # Skills to emphasize
        }
        
        # Learning and history
        self.learning_rate = random.uniform(0.05, 0.15)  # How quickly the agent adapts
        self.application_history = []
        self.interview_history = []
        self.offer_history = []
        self.reward_history = []
        
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
            skills=", ".join([f"{k}: {v}" for k, v in self.profile.get("skills", {}).items()]),
            risk_tolerance=self.risk_tolerance,
            truth_level=self.truth_level,
            interview_style=self.interview_style,
            interview_attitude=self.interview_attitude,
            lyingness=self.lyingness,
            skills_to_improve=", ".join(self.skills_to_improve)
        )
    
    def evaluate_job_listings(self, job_listings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to select which jobs to apply for.
        
        Args:
            job_listings: List of available job listings
            
        Returns:
            List of selected job listings
        """
        # If no jobs available, return empty list
        if not job_listings:
            return []
            
        # Create job selector agent
        job_selector = autogen.AssistantAgent(
            name=f"job_selector_{self.id}",
            llm_config=self.llm_config,
            system_message=f"""
            You are helping Candidate {self.id} select job listings to apply for.
            
            The candidate has these skills: {json.dumps(self.profile.get('skills', {}))}
            Sector preferences: {json.dumps(self.profile.get('motivation', {}))}
            Money importance: {self.profile.get('money_importance', 0.5)}
            """
        )
        
        user_proxy = create_user_proxy(name="job_selection_user")
        
        # Concatenate all job listings
        job_titles_text = ""
        for i, job in enumerate(job_listings):
            title = job.get("title", "Data Scientist")
            company_name = job.get("company_name", f"Company {job.get('company_id')}")
            job_titles_text += f"{i+1}. {title} at {company_name}\n"
        
        # Determine max applications
        max_applications = min(
            self.profile.get("energy", INITIAL_ENERGY),
            MAX_APPLICATIONS_PER_CANDIDATE
        )
        
        # Ask LLM to select jobs - very simple output
        user_proxy.initiate_chat(
            job_selector,
            message=f"""
            Here are available jobs:
            
            {job_titles_text}
            
            Select up to {max_applications} jobs for the candidate to apply to.
            
            ONLY respond with bullet points of job numbers you select. Example:
            * 1
            * 3
            
            No explanations or other text - just the bullet list of numbers.
            """
        )
        
        # Extract selected jobs from response
        response = job_selector.last_message()["content"]
        selected_job_indices = []
        
        # Parse response to find job numbers
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("*") or line.startswith("-"):
                # Extract job number
                numbers = re.findall(r'\d+', line)
                if numbers:
                    job_idx = int(numbers[0]) - 1  # Convert to 0-based index
                    if 0 <= job_idx < len(job_listings):
                        selected_job_indices.append(job_idx)
        
        # Get selected jobs
        selected_jobs = [job_listings[i] for i in selected_job_indices]
        
        # If parsing failed, select top jobs based on sector preference
        if not selected_jobs and job_listings:
            sector_prefs = self.profile.get("motivation", {})
            sorted_jobs = sorted(
                range(len(job_listings)), 
                key=lambda i: sector_prefs.get(job_listings[i].get("sector", ""), 0),
                reverse=True
            )
            selected_jobs = [job_listings[i] for i in sorted_jobs[:max_applications]]
        
        print(f"Candidate {self.id} selected {len(selected_jobs)} jobs to apply for")
        return selected_jobs
    
    def create_cv(self, truth_level: float = 1.0) -> Dict[str, Any]:
        """
        Create a CV, potentially with exaggerations, based on the candidate's full profile.
        
        Args:
            truth_level: How truthful to be (1.0 = completely honest)
            
        Returns:
            CV data
        """
        
        # Create resume generator
        resume_generator = autogen.AssistantAgent(
            name=f"resume_generator_{self.id}",
            llm_config=self.llm_config,
            system_message=f"""
            You create resumes for job applications based on candidate profiles.
            Generate content that aligns with a recent graduate from a French university.
            """
        )
        
        user_proxy = create_user_proxy(name="resume_user")
        
        # Get job details from the most recent application
        job_id = None
        job_title = "Data Scientist"
        company_name = "Company"
        required_skills = []
        company_sector = "Technology"
        
        if self.application_history:
            latest_app = self.application_history[-1]
            job_listing = latest_app.get("job_listing", {})
            job_title = job_listing.get("title", job_title)
            company_name = job_listing.get("company_name", company_name)
            required_skills = job_listing.get("required_skills", [])
            company_sector = job_listing.get("sector", company_sector)
            job_id = job_listing.get("id")
        
        # Calculate truth level based on risk tolerance and strategy
        local_truth_level = max(0.5, self.truth_level - (0.05 * self.risk_tolerance))
        
        # Generate skills with potential exaggeration
        skills = {}
        for skill, level in self.profile.get("skills", {}).items():
            # Determine if this skill should be exaggerated
            should_exaggerate = False
            
            # Exaggerate if it's a required skill and truthfulness check fails
            if skill in required_skills and random.random() > local_truth_level:
                should_exaggerate = True
            
            # Exaggerate if it's a skill the candidate wants to emphasize
            if skill in self.application_strategy.get("skill_emphasis", []) and random.random() > local_truth_level:
                should_exaggerate = True
                
            if should_exaggerate:
                # Exaggerate skill level (more exaggeration for higher risk tolerance)
                exaggeration = random.randint(1, int(1 + self.risk_tolerance * 3))
                skills[skill] = min(level + exaggeration, 10)
            else:
                skills[skill] = level
        
        # Get education details
        education = self.profile.get("education", {})
        university = education.get("university", "University")
        degree = education.get("degree", "Master's")
        field = education.get("field", "Data Science")
        
        # Career goals
        career_goals = self.profile.get("career_goals", ["Technical expertise", "Work-life balance"])
        
        # Get sector motivation for targeted resume customization
        sector_motivation = self.profile.get("motivation", {}).get(company_sector, 5)
        
        # Tailor resume emphasis based on sector and motivation
        sector_emphasis = ""
        if sector_motivation > 7:
            sector_emphasis = f"The candidate is highly motivated to work in the {company_sector} sector. Emphasize relevant experience or coursework."
        
        # Determine work experience level based on risk tolerance and truthfulness
        work_exp_count = 2  # Default
        if self.risk_tolerance > 0.6 and local_truth_level < 0.8:
            # Higher risk candidates with lower truth might add more experience
            work_exp_count = random.randint(3, 4)
        else:
            work_exp_count = random.randint(2, 3)
        
        # Ask LLM to create resume
        user_proxy.initiate_chat(
            resume_generator,
            message=f"""
            Create a resume for a recent graduate applying for a {job_title} position at {company_name} in the {company_sector} sector.
            
            Candidate profile:
            - Education: {degree} in {field} from {university}
            - Career goals: {', '.join(career_goals)}
            - Skills: {json.dumps(skills)}
            - Work-life balance importance: {self.profile.get('work_life_balance', 0.5)}
            
            {sector_emphasis}
            
            Include these sections:
            1. EDUCATION (including the real education from above)
            2. PROFESSIONAL EXPERIENCE ({work_exp_count} entries suitable for a recent graduate)
            3. SKILLS (incorporate the skills provided above)
            4. PROJECTS (2-3 relevant projects)
            5. HOBBIES (2-3 entries)
            
            Format as JSON with these exact keys: "education", "professional_experience", "skills", "projects", "hobbies".
            
            For professional experience, create entries appropriate for a student or recent graduate (internships, part-time work, research assistantships).
            """
        )
        
        # Extract resume content
        response = resume_generator.last_message()["content"]
        
        # Parse JSON content
        try:
            # Find and extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                # Try to find JSON directly
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                else:
                    raise ValueError("No JSON found in response")
                    
            resume_content = json.loads(json_str)
        except Exception as e:
            print(f"Error parsing resume JSON for Candidate {self.id}: {e}")
            # Create basic fallback resume with real education
            resume_content = {
                "education": [
                    {"degree": degree, "institution": university, "field": field, "year": "2020-2023"}
                ],
                "professional_experience": [
                    {
                        "title": "Data Science Intern", 
                        "company": f"{company_sector} Solutions", 
                        "period": "2022-2023",
                        "responsibilities": ["Data analysis", "Reporting", "Dashboard creation"]
                    }
                ],
                "skills": skills,
                "projects": [
                    {"name": "Data Analysis Project", "description": "Academic project analyzing public datasets"}
                ],
                "hobbies": ["Programming", "Data Visualization"]
            }
        
        # Calculate expected salary based on money importance, career goals, and market
        money_importance = self.profile.get("money_importance", 0.5)
        # French junior data scientist salary range adjustment
        base_min = 30000
        base_max = 45000
        
        # Adjust for education level
        if degree == "PhD":
            base_min += 5000
            base_max += 8000
        elif degree == "Master's":
            base_min += 2000
            base_max += 4000
            
        # Calculate expected salary
        expected_salary = random.randint(
            int(base_min * (1 + 0.1 * money_importance)),
            int(base_max * (0.8 + 0.2 * money_importance))
        )
        
        # Add metadata about exaggerations for internal tracking
        exaggerated_skills = {k: v for k, v in skills.items() if v > self.profile.get("skills", {}).get(k, 0)}
        
        # Create final CV
        cv_data = {
            "candidate_id": self.id,
            "job_id": job_id,
            "company_id": None,  # Will be filled in when applying
            "content": resume_content,
            "skills": skills,
            "truth_level": local_truth_level,
            "expected_salary": expected_salary,
            "target_skills": required_skills,
            "_meta": {
                "exaggerated_skills": exaggerated_skills,
                "sector_motivation": sector_motivation,
                "career_goals": career_goals
            }
        }
        
        return cv_data
    
    def prepare_for_interview(self, job_listing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategically prepare for an interview.
        
        Args:
            job_listing: Job listing for the interview
            
        Returns:
            Interview preparation data
        """
        # Use the predefined interview style and attitude
        strategy = self.interview_style
        attitude = self.interview_attitude
        
        # Consume energy
        self.profile["energy"] = max(0, self.profile.get("energy", 0) - 1)
        
        return {
            "candidate_id": self.id,
            "job_id": job_listing.get("id"),
            "company_id": job_listing.get("company_id"),
            "strategy": strategy,
            "attitude": attitude,
            "energy": self.profile["energy"],
            "lyingness": self.lyingness,
            "skills_to_improve": self.skills_to_improve
        }
    
    def rank_job_offers(self, offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to rank job offers.
        
        Args:
            offers: List of job offers received
            
        Returns:
            Ranked list of offers with acceptance decision
        """
        if not offers:
            return []
        
        # Create offer evaluator
        evaluator = autogen.AssistantAgent(
            name=f"offer_evaluator_{self.id}",
            llm_config=self.llm_config,
            system_message=f"""
            You are ranking job offers for a candidate.
            """
        )
        
        user_proxy = create_user_proxy(name="offer_evaluation_user")
        
        # Format offers
        offers_text = ""
        for i, offer in enumerate(offers):
            company_id = offer.get("company_id", "Unknown")
            job_id = offer.get("job_id", "Unknown")
            salary = offer.get("salary", "Not specified")
            
            # Find job title if possible
            job_title = "Data Scientist"
            for app in self.application_history:
                if app.get("job_id") == job_id and app.get("company_id") == company_id:
                    if app.get("job_listing", {}).get("title"):
                        job_title = app.get("job_listing", {}).get("title")
                    break
            
            offers_text += f"{i+1}. {job_title} at Company {company_id}, Salary: ${salary}\n"
        
        # Ask LLM to rank offers - simple output
        user_proxy.initiate_chat(
            evaluator,
            message=f"""
            Rank these job offers:
            
            {offers_text}
            
            ONLY respond with two lines:
            RANKING: comma-separated list of offer numbers in preference order
            ACCEPT: which offer number to accept (or NONE)
            
            Example:
            RANKING: 2,1,3
            ACCEPT: 2
            """
        )
        
        # Extract results
        response = evaluator.last_message()["content"]
        ranking = []
        accepted_idx = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('RANKING:'):
                # Extract numbers from ranking
                nums = re.findall(r'\d+', line)
                ranking = [int(num) - 1 for num in nums if 0 <= int(num) - 1 < len(offers)]
            elif line.startswith('ACCEPT:'):
                # Extract accepted offer
                nums = re.findall(r'\d+', line)
                if nums:
                    accepted_idx = int(nums[0]) - 1
                    if not (0 <= accepted_idx < len(offers)):
                        accepted_idx = None
        
        # Default to order of offers if no ranking returned
        if not ranking and offers:
            ranking = list(range(len(offers)))
        
        # Default to accepting best offer if no acceptance specified
        if accepted_idx is None and ranking:
            accepted_idx = ranking[0]
        
        # Create ranked list with acceptance decision
        ranked_offers = []
        for idx in ranking:
            offer = offers[idx].copy()
            offer["accepted"] = (idx == accepted_idx)
            ranked_offers.append(offer)
        
        # Add any missing offers
        included_indices = set(ranking)
        for i, offer in enumerate(offers):
            if i not in included_indices:
                offer_copy = offer.copy()
                offer_copy["accepted"] = False
                ranked_offers.append(offer_copy)
        
        return ranked_offers
    
    def decide_on_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision on a job offer.
        
        Args:
            offer: Job offer to consider
            
        Returns:
            Decision data
        """
        # Calculate minimum acceptable salary
        min_acceptable_salary = MIN_SALARY * (1 + 0.5 * self.profile.get("money_importance", 0.5))
        
        # Get offered salary
        salary = offer.get("salary", 0)
        
        # Decision is based on whether salary meets minimum and strategic considerations
        accepted = salary >= min_acceptable_salary
        
        # If multiple offers, might be more selective
        if len(self.offer_history) > 0 and len([o for o in self.offer_history if o.get("status") == "pending"]) > 1:
            # With multiple offers, be more selective
            accepted = salary >= min_acceptable_salary * 1.2
        
        decision = {
            "candidate_id": self.id,
            "job_id": offer.get("job_id"),
            "company_id": offer.get("company_id"),
            "accepted": accepted,
            "reason": "Salary meets expectations" if accepted else "Salary below expectations"
        }
        
        return decision
    
    def adjust_strategy(self, rewards: float, outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy based on outcomes and rewards.
        
        Args:
            rewards: Reward value received
            outcomes: Outcome details
            
        Returns:
            Dictionary with strategy changes
        """
        # Record history
        self.reward_history.append(rewards)
        
        # No adjustment if no history
        if len(self.reward_history) < 2:
            return {}
            
        # Check if rewards improved
        reward_improved = self.reward_history[-1] > self.reward_history[-2]
        
        # Extract specific outcomes - handle both enhanced and legacy structures
        # Enhanced structure
        if "candidates" in outcomes and self.id in outcomes["candidates"]:
            candidate_outcomes = outcomes["candidates"][self.id]
            lying_detected = candidate_outcomes.get("lying_detected", False)
            accepted_offer = candidate_outcomes.get("accepted_offer", None)
            interviewed = len(candidate_outcomes.get("interviewed_for", [])) > 0
            shortlisted = len(candidate_outcomes.get("shortlisted_for", [])) > 0
            offers_count = candidate_outcomes.get("offers_count", 0)
        else:
            # Legacy structure
            lying_detected = outcomes.get("detected_lying", False)
            accepted_offer = outcomes.get("offer_accepted", False)
            interviewed = outcomes.get("interviewed", False)
            shortlisted = outcomes.get("application_accepted", False)
            offers_count = 1 if outcomes.get("received_offer", False) else 0
        
        # Record strategic changes
        strategic_changes = {}
        
        # Adjust truth level and lyingness based on outcomes
        if lying_detected:
            previous_lyingness = self.lyingness
            self.lyingness = max(0.0, self.lyingness - 0.3)
            strategic_changes["lyingness"] = {
                "before": previous_lyingness,
                "after": self.lyingness,
                "reason": "lying_detected"
            }
            print(f"Candidate {self.id} was caught lying and reduced lyingness from {previous_lyingness:.2f} to {self.lyingness:.2f}")
        elif accepted_offer:
            # If successful with honesty, maintain it
            if self.truth_level > 0.8 and reward_improved:
                # Honesty worked well, slightly reinforce it
                previous_truth = self.truth_level
                self.truth_level = min(1.0, self.truth_level + 0.05)
                strategic_changes["truth_level"] = {
                    "before": previous_truth,
                    "after": self.truth_level,
                    "reason": "honesty_rewarded"
                }
            elif self.truth_level < 0.8 and reward_improved:
                # Some dishonesty worked, maintain it
                pass
            elif not reward_improved:
                # Adjust based on risk tolerance
                if self.risk_tolerance > 0.6:
                    # High-risk candidates might try something different
                    if random.random() < 0.5:  # 50% chance
                        prev_interview_style = self.interview_style
                        self.interview_style = "Formal" if self.interview_style == "Casual" else "Casual"
                        strategic_changes["interview_style"] = {
                            "before": prev_interview_style,
                            "after": self.interview_style,
                            "reason": "trying_different_approach"
                        }
                else:
                    # Low-risk candidates stick to more honesty
                    previous_truth = self.truth_level
                    self.truth_level = min(1.0, self.truth_level + 0.05)
                    strategic_changes["truth_level"] = {
                        "before": previous_truth,
                        "after": self.truth_level,
                        "reason": "being_more_honest"
                    }
        elif offers_count > 0:
            # Got offers but didn't accept any
            previous_truth = self.truth_level
            self.truth_level = min(1.0, self.truth_level + 0.05)
            strategic_changes["truth_level"] = {
                "before": previous_truth,
                "after": self.truth_level,
                "reason": "offers_received_but_declined"
            }
        elif interviewed:
            # Got interviewed but no offers
            if self.risk_tolerance > 0.6:
                # High risk candidates might try more strategic approach
                if random.random() < 0.4:  # 40% chance
                    previous_lyingness = self.lyingness
                    self.lyingness = min(0.5, self.lyingness + 0.05)
                    strategic_changes["lyingness"] = {
                        "before": previous_lyingness,
                        "after": self.lyingness,
                        "reason": "interviews_no_offers_high_risk"
                    }
                else:
                    # Or they might change interview attitude
                    prev_attitude = self.interview_attitude
                    self.interview_attitude = "Humble" if self.interview_attitude == "Confident" else "Confident"
                    strategic_changes["interview_attitude"] = {
                        "before": prev_attitude,
                        "after": self.interview_attitude,
                        "reason": "interviews_no_offers_changing_attitude"
                    }
            else:
                # Low risk candidates might try more honesty
                previous_truth = self.truth_level
                self.truth_level = min(1.0, self.truth_level + 0.05)
                strategic_changes["truth_level"] = {
                    "before": previous_truth,
                    "after": self.truth_level,
                    "reason": "interviews_no_offers_low_risk"
                }
        elif shortlisted:
            # Was shortlisted but not interviewed
            if self.risk_tolerance > 0.5:
                # Moderately adapt interview style
                prev_style = self.interview_style
                self.interview_style = "Formal" if self.interview_style == "Casual" else "Casual"
                strategic_changes["interview_style"] = {
                    "before": prev_style,
                    "after": self.interview_style,
                    "reason": "shortlisted_no_interview_changing_style"
                }
            else:
                # No major changes for low-risk candidates
                pass
        else:
            # No progress at all
            if self.risk_tolerance > 0.7:
                # Very high risk candidates might become more strategic
                previous_lyingness = self.lyingness
                self.lyingness = min(0.4, self.lyingness + 0.1)
                strategic_changes["lyingness"] = {
                    "before": previous_lyingness,
                    "after": self.lyingness,
                    "reason": "desperate_high_risk"
                }
            elif self.risk_tolerance > 0.4:
                # Medium risk candidates might emphasize different skills
                self.application_strategy["skill_emphasis"] = random.sample(
                    list(self.profile.get("skills", {}).keys()),
                    min(3, len(self.profile.get("skills", {})))
                )
                strategic_changes["skill_emphasis"] = {
                    "new_emphasis": self.application_strategy["skill_emphasis"],
                    "reason": "changing_focus"
                }
            else:
                # Low risk candidates stay consistent
                pass
        
        # Update skills to improve based on successful skills
        successful_skills = outcomes.get("successful_skills", [])
        if successful_skills:
            # Find skills with lowest proficiency to improve
            current_skills = self.profile.get("skills", {})
            new_skills_to_improve = sorted(
                successful_skills,
                key=lambda skill: current_skills.get(skill, 0)
            )[:2]  # Take up to 2 skills to improve
            
            if set(new_skills_to_improve) != set(self.skills_to_improve):
                strategic_changes["skills_to_improve"] = {
                    "before": self.skills_to_improve,
                    "after": new_skills_to_improve,
                    "reason": "focusing_on_successful_skills"
                }
                self.skills_to_improve = new_skills_to_improve
        
        # Improve skills that were targeted for improvement
        current_skills = self.profile.get("skills", {})
        skills_improved = []
        for skill in self.skills_to_improve:
            if skill in current_skills and current_skills[skill] < 10:
                # Improve skill by 1 point (up to max of 10)
                old_level = current_skills[skill]
                current_skills[skill] = min(10, current_skills[skill] + 1)
                if old_level != current_skills[skill]:
                    skills_improved.append(skill)
                    print(f"Candidate {self.id} improved {skill} skill to level {current_skills[skill]}")
        
        if skills_improved:
            strategic_changes["skills_improved"] = skills_improved
        
        return strategic_changes


class CompanyAgent:
    """Agent representing a company with strategic hiring practices."""
    
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
        self.llm_config = llm_config
        
        # Strategic parameters
        self.hiring_standards = random.uniform(0.5, 0.9)  # How high the bar is
        self.skill_emphasis = random.sample(
            list(self.profile.get("required_skills", [])),  # Already a list, but ensure it
            min(2, len(self.profile.get("required_skills", [])))
        )   # Skills to emphasize in hiring
        self.budget_flexibility = random.uniform(0.1, 0.3)  # How much over budget they'll go
        
        # Learning and history
        self.learning_rate = random.uniform(0.05, 0.15)  # How quickly the company adapts
        self.job_listing_history = []
        self.application_history = []
        self.interview_history = []
        self.hiring_history = []
        self.reward_history = []
        self.offer_history = []
        
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
            skills=", ".join(self.profile.get("required_skills", [])),
            hiring_standards=self.hiring_standards,
            budget_flexibility=self.budget_flexibility
        )
    
    def create_job_listing(self) -> Dict[str, Any]:
        """
        Create a strategic job listing.
        
        Returns:
            Job listing data
        """
        # Build on past successful listings if available
        successful_skills = set()
        if self.hiring_history:
            for hire in self.hiring_history:
                if hire.get("success", False):
                    job_id = hire.get("job_id")
                    for listing in self.job_listing_history:
                        if listing.get("id") == job_id:
                            successful_skills.update(listing.get("required_skills", []))
        
        # If we have successful skills, emphasize them
        required_skills = self.profile.get("required_skills", [])
        if successful_skills:
            # Ensure at least some successful skills are included
            overlap = list(set(required_skills) & successful_skills)
            if overlap:
                # Replace some skills with successful ones
                required_skills = list(set(required_skills) - set(random.sample(required_skills, min(2, len(required_skills)))))
                required_skills.extend(random.sample(list(successful_skills), min(2, len(successful_skills))))
        
        # Generate salary range based on budget and flexibility
        budget = self.profile.get("budget", 100000)
        min_salary = int(budget * 0.8)
        max_salary = int(budget * (1 + self.budget_flexibility))
        
        # Create listing
        listing = {
            "id": len(self.job_listing_history) + 1,
            "company_id": self.id,
            "company_name": self.profile.get("name", f"Company {self.id}"),
            "sector": self.profile.get("sector", "Technology"),
            "required_skills": required_skills,
            "salary_range": (min_salary, max_salary),
            "hiring_standards": self.hiring_standards,
            "emphasis_skills": self.skill_emphasis.copy()
        }
        
        # Store in history
        self.job_listing_history.append(listing)
        
        return listing
    
    def evaluate_applications(self, applications: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Evaluate candidate applications strategically.
        
        Args:
            applications: List of applications
            
        Returns:
            List of (application, score) tuples sorted by score
        """
        # Get weights
        weights = CANDIDATE_EVALUATION_WEIGHTS
        
        scored_applications = []
        for app in applications:
            # Verify application is for this company
            if app.get("company_id") != self.id:
                continue
                
            # Get candidate skills
            candidate_skills = app.get("skills", {})
            
            # Get required skills and emphasis skills
            required_skills = app.get("target_skills", [])
            emphasis_skills = self.skill_emphasis
            
            # Calculate skill match with emphasis on priority skills
            basic_skill_match = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 0
            emphasis_match = sum(candidate_skills.get(skill, 0) for skill in emphasis_skills) / len(emphasis_skills) if emphasis_skills else 0
            
            # Combined skill score
            skill_match = 0.7 * basic_skill_match + 0.3 * emphasis_match
            
            # Budget considerations
            budget = self.profile.get("budget", 0)
            expected_salary = app.get("expected_salary", budget)
            
            # More flexible budget score
            budget_fit = 1.0
            if expected_salary > budget:
                # How much over budget?
                over_budget = (expected_salary - budget) / budget
                if over_budget <= self.budget_flexibility:
                    # Within flexibility range
                    budget_fit = 1.0 - (over_budget / self.budget_flexibility)
                else:
                    # Beyond flexibility
                    budget_fit = 0.0
            
            # Apply hiring standards - higher standards = higher bar
            skill_threshold = 5 + (self.hiring_standards * 4)  # Scales from 7 to 8.6
            if skill_match * 10 < skill_threshold:  # Convert to 0-10 scale
                skill_match *= 0.5  # Penalize skills below threshold
            
            # Combine factors using weights
            score = (
                weights["skill_match"] * skill_match +
                weights["budget_fit"] * budget_fit
            )
            
            scored_applications.append((app, score))
        
        # Sort by score (descending)
        scored_applications.sort(key=lambda x: x[1], reverse=True)
        return scored_applications
    
    def shortlist_candidates(self, applications: List[Dict[str, Any]], num_slots: int = 2) -> List[Dict[str, Any]]:
        """
        Use LLM to select which candidates to shortlist for interviews.
        
        Args:
            applications: List of applications with resumes
            num_slots: Maximum number of interview slots
            
        Returns:
            List of shortlisted applications
        """
        # Filter applications for this company
        company_applications = [app for app in applications if app.get("company_id") == self.id]
        
        # If fewer applications than slots, shortlist all
        if len(company_applications) <= num_slots:
            return company_applications
        
        # Create shortlisting agent
        shortlister = autogen.AssistantAgent(
            name=f"shortlister_{self.id}",
            llm_config=self.llm_config,
            system_message=f"""
            You are ranking applicants for {self.profile.get('name', f'Company {self.id}')} in the {self.profile.get('sector', 'Technology')} sector.
            Required skills: {', '.join(self.profile.get('required_skills', []))}
            """
        )
        
        user_proxy = create_user_proxy(name="shortlist_user")
        
        # Create candidate list
        candidate_list = ""
        for i, app in enumerate(company_applications):
            resume = app.get("resume", {})
            resume_content = resume.get("content", {})
            candidate_id = app.get("candidate_id")
            
            # Add education and skills - keep it brief
            education = resume_content.get("education", [])
            education_text = f"Education: {education[0].get('degree', 'Degree')} from {education[0].get('institution', 'Institution')}" if education else "No education listed"
            
            skills_text = "Skills: " + ", ".join([f"{skill}: {level}" for skill, level in resume_content.get("skills", {}).items()])
            
            candidate_list += f"{i+1}. Candidate {candidate_id}: {education_text}. {skills_text}\n\n"
        
        # Ask LLM to rank candidates - extremely simple output
        user_proxy.initiate_chat(
            shortlister,
            message=f"""
            Rank these candidates for the position:
            
            {candidate_list}
            
            ONLY respond with the ranking as a list of candidate numbers in order. Example:
            1. 3
            2. 1
            3. 2
            
            No explanations - just the numbered ranking.
            """
        )
        
        # Extract ranking from response
        response = shortlister.last_message()["content"]
        ranked_indices = []
        
        # Parse response to find candidate rankings
        for line in response.split("\n"):
            line = line.strip()
            if re.match(r'^\d+\.', line):  # Matches lines starting with a number followed by a period
                # Extract candidate number
                match = re.search(r'(\d+)$', line)  # Find number at end of line
                if match:
                    candidate_idx = int(match.group(1)) - 1  # Convert to 0-based index
                    if 0 <= candidate_idx < len(company_applications):
                        ranked_indices.append(candidate_idx)
        
        # Get the top candidates up to num_slots
        shortlist = [company_applications[i] for i in ranked_indices[:num_slots] if i < len(company_applications)]
        
        # If parsing failed, use simple criteria
        if not shortlist and company_applications:
            # Sort by expected salary (lower is better for company)
            shortlist = sorted(
                company_applications,
                key=lambda app: app.get("resume", {}).get("expected_salary", MAX_SALARY)
            )[:num_slots]
        
        print(f"Company {self.id} shortlisted {len(shortlist)} candidates for interviews")
        return shortlist
    
    def design_interview_process(self, candidate_application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a customized interview process for a candidate.
        
        Args:
            candidate_application: Candidate's application
            
        Returns:
            Interview process data
        """
        # Choose strategy based on skills and sector
        candidate_skills = candidate_application.get("skills", {})
        required_skills = candidate_application.get("target_skills", [])
        
        # Determine interview complexity based on skill level
        avg_skill_level = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 5
        
        # Higher level candidates get more rigorous interviews
        if avg_skill_level > 7:
            interview_style = "Technical"
            num_rounds = random.randint(3, 5)  # More rounds for highly skilled candidates
        else:
            interview_style = "Behavioral"
            num_rounds = random.randint(2, 3)
        
        # Adapt to sector norms
        sector = self.profile.get("sector", "Technology")
        if sector in ["Finance", "Healthcare"]:
            attitude = "Formal"
        else:
            attitude = "Casual"
        
        return {
            "company_id": self.id,
            "candidate_id": candidate_application.get("candidate_id"),
            "job_id": candidate_application.get("job_id"),
            "style": interview_style,
            "attitude": attitude,
            "num_rounds": num_rounds,
            "focus_skills": self.skill_emphasis
        }
    
    def evaluate_interview(self, interview_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to evaluate an interview result and detect lying.
        
        Args:
            interview_result: Results from the interview
            
        Returns:
            Dictionary with evaluation results
        """
        evaluator = autogen.AssistantAgent(
            name=f"interview_evaluator_{self.id}",
            llm_config=self.llm_config,
            system_message=f"""
            You are evaluating a job interview for a {self.profile.get('sector', 'Technology')} company.
            
            Analyze both the technical competence and the honesty of the candidate.
            Look for inconsistencies or exaggerations in their claims versus their demonstrated knowledge.
            """
        )
        
        user_proxy = create_user_proxy(name="interview_evaluation_user")
        
        # Extract interview messages
        messages = interview_result.get("messages", [])
        
        # Format the conversation
        interview_text = ""
        for msg in messages:
            sender = msg.get("name", "Unknown")
            content = msg.get("content", "")
            interview_text += f"{sender}: {content}\n\n"
        
        # Ask LLM to evaluate - output score and lie detection
        user_proxy.initiate_chat(
            evaluator,
            message=f"""
            Evaluate this interview:
            
            {interview_text}
            
            Please respond with these two lines:
            SCORE: [number between 0-10]
            LYING_DETECTED: [YES/NO]
            
            Make your determination about lying based on inconsistencies between the candidate's claimed skills/experience and their answers to technical questions.
            """
        )
        
        # Extract evaluation
        response = evaluator.last_message()["content"]
        
        score = 0.5  # Default
        lying_detected = False
        
        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+(\.\d+)?)', response)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(10.0, score)) / 10.0  # Normalize to 0-1
        
        # Extract lying detection
        lying_match = re.search(r'LYING_DETECTED:\s*(YES|NO)', response)
        if lying_match:
            lying_detected = lying_match.group(1) == "YES"
        
        # If lying detected, apply penalty to score
        if lying_detected:
            score *= 0.5  # Significant penalty for lying
            print(f"Lying detected in interview for Candidate {interview_result.get('candidate_id')}, score penalized")
        
        print(f"Interview evaluation score: {score:.2f}, Lying detected: {lying_detected}")
        
        return {
            "score": score,
            "lying_detected": lying_detected
        }
    
    def rank_interviewed_candidates(self, interviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank candidates based on interviews.
        
        Args:
            interviews: List of interview results
            
        Returns:
            Ranked list of candidates
        """
        # Look up scores from interview history
        scored_candidates = []
        for interview in interviews:
            interview_id = interview.get("interview_id")
            
            # Find score in history
            score = 0.0
            for record in self.interview_history:
                if record.get("interview_id") == interview_id:
                    score = record.get("score", 0.0)
                    break
            
            scored_candidates.append((interview, score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked interviews
        return [interview for interview, _ in scored_candidates]
    
    def make_job_offer(self, candidate_id: int, job_id: int) -> Dict[str, Any]:
        """
        Create a job offer for a candidate.
        
        Args:
            candidate_id: Candidate ID
            job_id: Job ID
            
        Returns:
            Job offer data
        """
        # Find candidate application
        candidate_application = None
        for app_data in self.application_history:
            app = app_data.get("application", {})
            if app.get("candidate_id") == candidate_id and app.get("job_id") == job_id:
                candidate_application = app
                break
        
        if not candidate_application:
            return None
            
        # Find job listing
        job_listing = None
        for listing in self.job_listing_history:
            if listing.get("id") == job_id:
                job_listing = listing
                break
        
        if not job_listing:
            return None
            
        # Get salary range
        min_salary, max_salary = job_listing.get("salary_range", (MIN_SALARY, MAX_SALARY))
        
        # Get candidate's expected salary
        expected_salary = candidate_application.get("expected_salary", min_salary)
        
        # Strategic salary offer
        # If candidate skills are excellent, offer closer to max
        candidate_skills = candidate_application.get("skills", {})
        required_skills = job_listing.get("required_skills", [])
        
        avg_skill_level = sum(candidate_skills.get(skill, 0) for skill in required_skills) / len(required_skills) if required_skills else 5
        skill_factor = avg_skill_level / 10  # 0-1 scale
        
        # Base offer adjusts with skill level
        base_offer = min_salary + (max_salary - min_salary) * skill_factor
        
        # Final offer considers expected salary
        if expected_salary <= base_offer:
            # If candidate expects less than we'd offer, give a bit more than expected
            offer_salary = min(max_salary, expected_salary * 1.05)
        else:
            # If candidate expects more, consider flexibility
            if expected_salary <= (1 + self.budget_flexibility) * max_salary:
                # Within flexibility range
                offer_salary = expected_salary
            else:
                # Beyond flexibility, offer maximum
                offer_salary = max_salary
        
        # Create offer
        offer = {
            "company_id": self.id,
            "candidate_id": candidate_id,
            "job_id": job_id,
            "salary": int(offer_salary)
        }

        self.offer_history.append(offer)
        
        return offer
    
    def make_hiring_decision(self, interview_result: Dict[str, Any]) -> bool:
        """
        Make a hiring decision based on an interview and whether lying was detected.
        
        Args:
            interview_result: Results from the interview
            
        Returns:
            Whether to hire the candidate
        """
        # Get evaluation
        interview_id = interview_result.get("interview_id")
        
        # Find evaluation in history
        score = 0.5  # Default
        lying_detected = False
        
        for record in self.interview_history:
            if record.get("interview_id") == interview_id:
                score = record.get("score", 0.5)
                lying_detected = record.get("lying_detected", False)
                break
        
        # Automatic rejection if lying detected
        if lying_detected:
            return False
        
        # Otherwise, use the score threshold
        return score >= self.hiring_standards
    
    def process_candidate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a candidate's decision on a job offer.
        
        Args:
            decision: Candidate decision
            
        Returns:
            Processed decision with company reaction
        """
        # Record in hiring history
        hiring_record = {
            "job_id": decision.get("job_id"),
            "candidate_id": decision.get("candidate_id"),
            "success": decision.get("accepted", False),
            "date": time.time()
        }
        self.hiring_history.append(hiring_record)
        
        # Add company reaction
        result = decision.copy()
        
        if decision.get("accepted", False):
            result["company_reaction"] = "Positive"
        else:
            result["company_reaction"] = "Disappointed"
        
        return result
    
    def adjust_strategy(self, rewards: float, outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust hiring strategy based on outcomes and rewards.
        
        Args:
            rewards: Reward value received
            outcomes: Outcome details
            
        Returns:
            Dictionary with strategy changes
        """
        # Record history
        self.reward_history.append(rewards)
        
        # No adjustment if no history
        if len(self.reward_history) < 2:
            return {}
            
        # Check if rewards improved
        reward_improved = self.reward_history[-1] > self.reward_history[-2]
        
        # Extract outcomes - handle both enhanced and legacy structures
        # Enhanced structure
        if "companies" in outcomes and self.id in outcomes["companies"]:
            company_outcomes = outcomes["companies"][self.id]
            hired = company_outcomes.get("hired")
            all_candidates_refused = company_outcomes.get("all_candidates_refused", False)
            no_qualified_candidates = company_outcomes.get("no_qualified_candidates", False)
            applicants_count = len(company_outcomes.get("applicants", []))
            offers_extended = len(company_outcomes.get("offers_extended", []))
        else:
            # Legacy structure
            hired = outcomes.get("candidate_hired", False) or outcomes.get("candidate_accepted_offer", False)
            all_candidates_refused = outcomes.get("all_candidates_refused", False)
            no_qualified_candidates = outcomes.get("no_qualified_candidates", False)
            applicants_count = outcomes.get("applicants_count", 0)
            offers_extended = outcomes.get("offers_extended", 0)
        
        # Record strategic changes
        strategic_changes = {}
        
        # Adjust hiring standards based on outcomes
        if not hired:
            if all_candidates_refused:
                # Significantly lower standards if company was too picky
                previous_standards = self.hiring_standards
                self.hiring_standards = max(0.4, self.hiring_standards - 0.15)
                strategic_changes["hiring_standards"] = {
                    "before": previous_standards,
                    "after": self.hiring_standards,
                    "reason": "all_candidates_refused"
                }
                print(f"Company {self.id} refused all candidates - lowering standards from {previous_standards:.2f} to {self.hiring_standards:.2f}")
                
                # Also increase budget flexibility
                previous_flexibility = self.budget_flexibility
                self.budget_flexibility = min(0.4, self.budget_flexibility + 0.1)
                strategic_changes["budget_flexibility"] = {
                    "before": previous_flexibility,
                    "after": self.budget_flexibility,
                    "reason": "all_candidates_refused"
                }
            elif no_qualified_candidates:
                # Moderately lower standards if no one qualified
                previous_standards = self.hiring_standards
                self.hiring_standards = max(0.45, self.hiring_standards - 0.1)
                strategic_changes["hiring_standards"] = {
                    "before": previous_standards,
                    "after": self.hiring_standards,
                    "reason": "no_qualified_candidates"
                }
                print(f"Company {self.id} found no qualified candidates - lowering standards from {previous_standards:.2f} to {self.hiring_standards:.2f}")
            elif applicants_count == 0:
                # Lower budget constraints if no applicants
                previous_flexibility = self.budget_flexibility
                self.budget_flexibility = min(0.35, self.budget_flexibility + 0.08)
                strategic_changes["budget_flexibility"] = {
                    "before": previous_flexibility,
                    "after": self.budget_flexibility,
                    "reason": "no_applicants"
                }
                print(f"Company {self.id} received no applications - increasing budget flexibility from {previous_flexibility:.2f} to {self.budget_flexibility:.2f}")
            elif offers_extended > 0:
                # Offers were made but none accepted - might be salary issue
                previous_flexibility = self.budget_flexibility
                self.budget_flexibility = min(0.4, self.budget_flexibility + 0.05)
                strategic_changes["budget_flexibility"] = {
                    "before": previous_flexibility,
                    "after": self.budget_flexibility,
                    "reason": "offers_rejected"
                }
        else:
            # Successfully hired a candidate
            if reward_improved:
                # If the current approach is working, maintain it
                pass
            else:
                # If reward didn't improve, make slight adjustments
                
                # If standards are quite high, maybe lower them slightly
                if self.hiring_standards > 0.7:
                    previous_standards = self.hiring_standards
                    self.hiring_standards = max(0.6, self.hiring_standards - 0.05)
                    strategic_changes["hiring_standards"] = {
                        "before": previous_standards,
                        "after": self.hiring_standards,
                        "reason": "fine_tuning_after_hire"
                    }
                
                # If budget flexibility is low, maybe increase it slightly
                if self.budget_flexibility < 0.2:
                    previous_flexibility = self.budget_flexibility
                    self.budget_flexibility = min(0.25, self.budget_flexibility + 0.03)
                    strategic_changes["budget_flexibility"] = {
                        "before": previous_flexibility,
                        "after": self.budget_flexibility,
                        "reason": "fine_tuning_after_hire"
                    }
        
        # Update skill emphasis based on successful skills
        successful_skills = outcomes.get("successful_skills", [])
        if successful_skills:
            previous_emphasis = self.skill_emphasis.copy()
            
            # Replace one emphasis skill with a successful one
            if self.skill_emphasis and successful_skills:
                new_skill = None
                for skill in successful_skills:
                    if skill not in self.skill_emphasis:
                        new_skill = skill
                        break
                
                if new_skill:
                    self.skill_emphasis.pop(0)
                    self.skill_emphasis.append(new_skill)
                    strategic_changes["skill_emphasis"] = {
                        "before": previous_emphasis,
                        "after": self.skill_emphasis,
                        "reason": "adapting_to_successful_hire"
                    }
        
        return strategic_changes