#!/usr/bin/env python3
"""
Entry point script to run the enhanced job matching simulation.
"""
import argparse
import json
import os
import sys
from datetime import datetime

# Ensure the module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from autogenjobmatch.simulation import run_simulation
from autogenjobmatch.config import NUM_CANDIDATES, NUM_COMPANIES, NUM_SIMULATION_ROUNDS
from autogenjobmatch.monitoring import init_agentops, end_agentops

def main():
    parser = argparse.ArgumentParser(description="Run the AutogenJobMatch simulation")
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES, 
                        help="Number of candidates")
    parser.add_argument("--companies", type=int, default=NUM_COMPANIES, 
                        help="Number of companies")
    parser.add_argument("--simulation-rounds", type=int, default=NUM_SIMULATION_ROUNDS, 
                        help="Number of simulation rounds")
    parser.add_argument("--output", type=str, default="results", 
                        help="Output directory")
    parser.add_argument("--disable-agentops", action="store_true", 
                        help="Disable AgentOps tracking")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed LLM outputs")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting simulation with {args.candidates} candidates and {args.companies} companies")
    print(f"Running for {args.simulation_rounds} simulation rounds")
    print(f"Verbose output: {'Enabled' if args.verbose else 'Disabled'}")
    
    # Set verbosity level
    os.environ["AUTOGEN_OUTPUT_VERBOSE"] = "1" if args.verbose else "0"
    
    # Initialize AgentOps - simplest approach
    init_agentops(disable=args.disable_agentops)
    
    try:
        # Run the simulation with new parameters
        results = run_simulation(
            num_candidates=args.candidates,
            num_companies=args.companies,
            num_rounds=args.simulation_rounds,
            output_dir=args.output
        )
        
        print(f"Simulation completed successfully")
        print(f"Results saved to {args.output}/simulation_{results['simulation_id']}.json")
        
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