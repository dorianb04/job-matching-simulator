#!/usr/bin/env python3
"""
Entry point script to run the job matching simulation.
"""
import argparse
import json
import os
import sys
from datetime import datetime

# Ensure the module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from autogenjobmatch.simulation import run_simulation
from autogenjobmatch.config import NUM_CANDIDATES, NUM_COMPANIES, DEFAULT_INTERVIEW_ROUNDS
from autogenjobmatch.monitoring import init_agentops, end_agentops

def main():
    parser = argparse.ArgumentParser(description="Run the AutogenJobMatch simulation")
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES, help="Number of candidates")
    parser.add_argument("--companies", type=int, default=NUM_COMPANIES, help="Number of companies")
    parser.add_argument("--rounds", type=int, default=DEFAULT_INTERVIEW_ROUNDS, help="Number of interview rounds")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--disable-agentops", action="store_true", help="Disable AgentOps tracking")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting simulation with {args.candidates} candidates and {args.companies} companies")
    print(f"Each interview will have {args.rounds} rounds")
    
    # Initialize AgentOps - simplest approach
    init_agentops(disable=args.disable_agentops)
    
    try:
        # Run the simulation
        results = run_simulation(
            num_candidates=args.candidates,
            num_companies=args.companies,
            interview_rounds=args.rounds
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output, f"simulation_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        # End AgentOps with success
        end_agentops("Success")
        
        return results
    except Exception as e:
        print(f"Simulation failed: {e}")
        
        # End AgentOps with failure
        end_agentops("Failed")
        
        # Re-raise the exception
        raise

if __name__ == "__main__":
    main()