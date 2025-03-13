"""
AgentOps integration for monitoring job matching simulation.
"""
import time
from typing import Dict, Any, Optional, List
import uuid
import agentops

class AgentOpsTracker:
    """Track simulation operations using AgentOps."""
    
    def __init__(self, api_key: Optional[str] = None, project_tag: str = "job-matching-sim"):
        """
        Initialize the AgentOps tracker.
        
        Args:
            api_key: AgentOps API key
            project_tag: Project tag for filtering
        """
        self.api_key = api_key
        self.start_time = time.time()
        self.enabled = False
        self.session_id = str(uuid.uuid4())  # Add session_id attribute
        
        # Initialize AgentOps if API key is provided
        if self.api_key:
            try:
                # Initialize with just the API key (correct syntax)
                agentops.init(api_key=self.api_key, default_tags=[project_tag])
                self.enabled = True
                print(f"✓ AgentOps tracking enabled (session: {self.session_id})")
                
                # Try to add session_id as a tag
                try:
                    agentops.add_tags([f"session:{self.session_id}"])
                except:
                    pass
                    
            except Exception as e:
                print(f"AgentOps initialization failed: {e}")
        else:
            print("AgentOps tracking disabled (no API key provided)")
    
    def track_event(self, event_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Track an event by adding a tag for it.
        
        Args:
            event_name: Name of the event
            metadata: Event metadata
        """
        if not self.enabled:
            return
            
        try:
            # Convert the event to a tag
            tags = [f"event:{event_name}"]
            
            # Add metadata as additional tags if appropriate
            if metadata:
                # Convert simple key-value pairs to tags
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        tags.append(f"{k}:{v}")
            
            # Add the tags to AgentOps
            agentops.add_tags(tags)
        except Exception as e:
            print(f"Error tracking event {event_name}: {e}")
    
    def track_agent_message(self, sender: str, content: str, receiver: Optional[str] = None) -> None:
        """
        Track a message between agents.
        
        Args:
            sender: Sender agent ID
            content: Message content
            receiver: Receiver agent ID (if applicable)
        """
        metadata = {
            "sender": sender,
            "content_length": len(content),
            "timestamp": time.time()
        }
        
        if receiver:
            metadata["receiver"] = receiver
            
        self.track_event("agent_message", metadata)
    
    def add_tags(self, tags: List[str]) -> None:
        """
        Add tags to the current session.
        
        Args:
            tags: List of tags to add
        """
        if not self.enabled:
            return
            
        try:
            agentops.add_tags(tags)
        except Exception as e:
            print(f"Error adding tags: {e}")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics by converting them to tags.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        if not self.enabled:
            return
            
        try:
            # Convert metrics to tags (key:value format)
            tags = [f"{k}:{v}" for k, v in metrics.items() 
                   if isinstance(v, (str, int, float, bool))]
            if tags:
                agentops.add_tags(tags)
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def end_session(self, success: bool = True, metadata: Dict[str, Any] = None) -> None:
        """
        End the tracking session.
        
        Args:
            success: Whether the simulation was successful
            metadata: Additional metadata
        """
        if not self.enabled:
            return
            
        try:
            # Add final tags for metadata
            if metadata:
                self.update_metrics(metadata)
            
            # End the session with appropriate status
            status = "Success" if success else "Failed"
            agentops.end_session(status)
            print(f"✓ AgentOps session ended with status: {status}")
        except Exception as e:
            print(f"Error ending session: {e}")