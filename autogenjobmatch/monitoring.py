"""
Minimal AgentOps integration based on official documentation.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

def init_agentops(api_key=None, disable=False):
    """
    Initialize AgentOps with the minimal approach from documentation.
    
    Args:
        api_key: Optional API key (overrides env variable)
        disable: Flag to disable AgentOps
    """
    if disable:
        print("AgentOps tracking disabled by user")
        return
        
    try:
        import agentops
        api_key = api_key or AGENTOPS_API_KEY
        
        if not api_key:
            print("No AgentOps API key found")
            return
            
        # Simple initialization according to docs - just 1 line
        agentops.init(api_key=api_key)
        print("✓ AgentOps tracking enabled")
    except Exception as e:
        print(f"AgentOps initialization failed: {e}")

def end_agentops(status="Success"):
    """
    End the AgentOps session with a status.
    
    Args:
        status: Session end status ('Success' or 'Failed')
    """
    try:
        import agentops
        agentops.end_session(status)
        print(f"✓ AgentOps session ended with status: {status}")
    except Exception as e:
        print(f"Failed to end AgentOps session: {e}")