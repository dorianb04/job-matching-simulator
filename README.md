# AutogenJobMatch

## Overview

AutogenJobMatch is a sophisticated multi-agent simulation framework that models the job market dynamics between candidates and companies. The simulation focuses on data science positions in France, creating realistic interactions between job seekers and employers throughout the hiring process.

The simulation leverages Autogen and large language models to create strategic agents that learn and adapt their behaviors based on outcomes. This project is particularly valuable for studying:

- Strategic behaviors in job markets
- Candidate decision-making and learning
- Company hiring practices and adaptation
- The effects of different interview strategies
- The impact of truthfulness in applications

## Key Features

- **Multi-agent simulation**: Realistic interactions between job candidates and companies
- **Strategic learning**: Agents adapt their strategies based on outcomes
- **Complete job cycle**: From job listings through applications, interviews, offers, and hiring
- **French job market focus**: Salary ranges and university backgrounds tailored to the French context
- **Detailed metrics**: Comprehensive statistics tracking for all simulation stages
- **AgentOps integration**: Optional monitoring and analytics for detailed performance tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autogenjobmatch.git
cd autogenjobmatch

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the root directory of the project with the following content:

```env
GROQ_API_KEY=your_groq_api_key_here
AGENTOPS_API_KEY=your_agentops_api_key_here  # Optional
```

You can adjust various simulation parameters in config.py, including:

- Number of candidates and companies
- Simulation rounds
- Salary ranges
- Skills and sectors
- Strategic parameters for agents

## Running the Simulation

### Basic Run
```bash
python run_simulation.py
```

This command will execute the simulation with default parameters and configurations.

### Advanced Options
```bash
python run_simulation.py --candidates 10 --companies 5 --simulation-rounds 3 --output custom_results
```
Parameters:

- `--candidates`: Number of job candidates (default: 5)
- `--companies`: Number of hiring companies (default: 3)
- `--simulation-rounds`: Number of simulation cycles (default: 3)
- `--output`: Directory for storing results (default: "results")
- `--disable-agentops`: Disable AgentOps tracking
- `--verbose`: Enable detailed output from LLM interactions

## Simulation Process
Each simulation round consists of the following steps:

1. Job Listing Creation: Companies create strategic job listings
2. Job Application: Candidates evaluate listings and apply to selected positions
3. Application Evaluation: Companies shortlist candidates based on their applications
4. Interviews: Companies conduct technical and behavioral interviews with shortlisted candidates
5. Hiring Decisions: Companies evaluate interviews and make offers
6. Offer Evaluation: Candidates rank and decide on received offers
7. Strategy Adjustment: Both parties adjust their strategies based on outcomes

## Agent Architecture

### Candidate Agents

Candidates are modeled with these strategic parameters:
- **Skills**: Technical abilities in various data science domains
- **Truth level**: Honesty in applications and interviews
- **Interview style**: Formal vs. casual approach
- **Interview attitude**: Humble vs. confident presentation
- **Risk tolerance**: Willingness to take chances
- **Learning rate**: How quickly they adapt strategies

### Company Agents

Companies are modeled with these strategic parameters:
- **Hiring standards**: Selectiveness in candidate evaluation
- **Budget flexibility**: Willingness to exceed planned compensation
- **Skill emphasis**: Priority skills for the position
- **Interview process**: Technical vs. behavioral focus
- **Learning rate**: How quickly they adapt hiring strategies

## Output and Analysis

Simulation results are saved to JSON files in the output directory, containing:
- Complete simulation configuration
- Round-by-round statistics
- Candidate and company profiles with strategy evolution
- Application, interview, and hiring outcomes
- Reward metrics for all agents

Example summary output:
```
===== SIMULATION SUMMARY =====
Total candidates hired: 10/15
Total applications: 42
Total interviews: 18
Total offers: 12
Rejected offers: 2
Average candidate reward: 0.72
Average company reward: 0.68

===== CANDIDATE STRATEGY EVOLUTION =====
Candidate 0:
  Initial interview style: Formal, Final: Casual
  Initial interview attitude: Humble, Final: Confident
  Initial lyingness: 0.15, Final: 0.10
  Skills improved: Python, Machine Learning
  Total rewards: 1.85
  Skill 'Python': 6 → 8
  Skill 'Machine Learning': 4 → 6
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Project initiated in the context of the course "Reinforcement Learning" of the Master in Data Sciences and Business Analytics at CentraleSupélec & ESSEC.
- Built with [Autogen](https://github.com/microsoft/autogen)
- Uses [Groq](https://groq.com/) for LLM inference
- Optional monitoring via [AgentOps](https://www.agentops.ai/)