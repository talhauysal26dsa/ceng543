"""
Base agent class for Multi-Agent RAG system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Multi-Agent RAG system.
    
    Each agent is responsible for a specific task in the RAG pipeline:
    - Retrieval: Finding relevant documents
    - Ranking: Re-ranking retrieved documents
    - Generation: Creating final answers
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for the agent
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            **kwargs: Additional keyword arguments
            
        Returns:
            AgentResult: Result of the processing
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        pass
    
    def log_result(self, result: AgentResult) -> None:
        """Log the result of agent execution."""
        if result.success:
            self.logger.info(f"Agent {self.name} completed successfully")
        else:
            self.logger.error(f"Agent {self.name} failed: {result.error}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update agent configuration."""
        self.config.update(new_config)
        self.logger.info(f"Updated configuration for agent {self.name}")
