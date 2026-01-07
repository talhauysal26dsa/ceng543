"""
Supervisor agent for Multi-Agent RAG system.
Coordinates the execution of multiple agents using LangChain supervisor pattern.
"""

from typing import Any, Dict, List, Optional, Callable
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentResult
from .retriever_agent import RetrieverAgent
from .ranker_agent import RankerAgent
from .summarizer_agent import SummarizerAgent

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """Task for an agent execution."""
    agent: BaseAgent
    input_data: Any
    status: AgentStatus = AgentStatus.PENDING
    result: Optional[AgentResult] = None
    error: Optional[str] = None


class Supervisor(BaseAgent):
    """
    Supervisor agent that coordinates multiple agents in the RAG pipeline.
    
    Uses LangChain supervisor pattern to orchestrate the execution of:
    1. Retriever Agent - Find relevant documents
    2. Ranker Agent - Re-rank documents by relevance
    3. Summarizer Agent - Generate final answer
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the supervisor agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("supervisor", config)
        
        # Initialize sub-agents
        self.agents = {}
        self.max_iterations = config.get("max_iterations", 10)
        self.timeout = config.get("timeout", 300)
        self.decision_threshold = config.get("decision_threshold", 0.8)
        
        # Initialize retriever agent
        retriever_config = config.get("retriever_agent", {})
        self.agents["retriever"] = RetrieverAgent(retriever_config)
        
        # Initialize ranker agent
        ranker_config = config.get("ranker_agent", {})
        self.agents["ranker"] = RankerAgent(ranker_config)
        
        # Initialize summarizer agent
        summarizer_config = config.get("summarizer_agent", {})
        self.agents["summarizer"] = SummarizerAgent(summarizer_config)
        
        # Define execution pipeline
        self.pipeline = ["retriever", "ranker", "summarizer"]
        
    def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Execute the complete RAG pipeline.
        
        Args:
            input_data: Query string or query object
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Final answer and metadata
        """
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="Invalid input data"
                )
            
            # Execute pipeline
            result = self._execute_pipeline(input_data, **kwargs)
            
            self.log_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Error in supervisor: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data=None,
                metadata={},
                error=error_msg
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for the pipeline.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid
        """
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0
        elif isinstance(input_data, dict):
            return "query" in input_data and len(input_data["query"].strip()) > 0
        return False
    
    def _execute_pipeline(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Execute the complete RAG pipeline.
        
        Args:
            input_data: Input query
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Final result
        """
        pipeline_results = {}
        current_data = input_data
        
        # Execute each agent in sequence
        for agent_name in self.pipeline:
            agent = self.agents[agent_name]
            
            self.logger.info(f"Executing {agent_name} agent")
            
            # Execute agent
            agent_result = agent.process(current_data, **kwargs)
            
            if not agent_result.success:
                return AgentResult(
                    success=False,
                    data=None,
                    metadata=pipeline_results,
                    error=f"Agent {agent_name} failed: {agent_result.error}"
                )
            
            # Store result
            pipeline_results[agent_name] = agent_result
            
            # Prepare input for next agent
            if agent_name == "retriever":
                # Pass retrieved documents to ranker
                current_data = {
                    "query": input_data if isinstance(input_data, str) else input_data["query"],
                    "documents": agent_result.data.documents
                }
            elif agent_name == "ranker":
                # Pass ranked documents to summarizer
                current_data = {
                    "query": input_data if isinstance(input_data, str) else input_data["query"],
                    "documents": agent_result.data.documents
                }
        
        # Return final result from summarizer
        final_result = pipeline_results["summarizer"]
        
        return AgentResult(
            success=True,
            data=final_result.data,
            metadata={
                "pipeline_results": pipeline_results,
                "execution_order": self.pipeline,
                "num_agents": len(self.pipeline)
            }
        )
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents."""
        return {name: "ready" for name in self.agents.keys()}
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "agents": list(self.agents.keys()),
            "pipeline_order": self.pipeline,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "decision_threshold": self.decision_threshold
        }
