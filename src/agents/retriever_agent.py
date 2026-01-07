"""
Retriever agent for Multi-Agent RAG system.
Combines BM25 and dense retrieval strategies.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult
from ..retrievers.bm25_retriever import BM25Retriever
from ..retrievers.dense_retriever import DenseRetriever
from ..retrievers.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class RetrieverAgent(BaseAgent):
    """
    Agent responsible for document retrieval using multiple strategies.
    
    Combines BM25 (sparse) and dense retrieval methods to find relevant
    documents for a given query.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retriever agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("retriever_agent", config)
        
        # Initialize retrievers based on configuration
        self.retrievers = {}
        self.max_documents = config.get("max_documents", 100)
        
        # Initialize BM25 retriever if configured
        if "bm25" in config.get("retrieval_strategies", []):
            bm25_config = config.get("bm25_config", {})
            self.retrievers["bm25"] = BM25Retriever(bm25_config)
            
        # Initialize dense retriever if configured
        if "dense" in config.get("retrieval_strategies", []):
            dense_config = config.get("dense_config", {})
            self.retrievers["dense"] = DenseRetriever(dense_config)
            
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever({
            "retrievers": self.retrievers,
            "max_documents": self.max_documents
        })
        
    def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            input_data: Query string or query object
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Retrieved documents and metadata
        """
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="Invalid input data"
                )
            
            # Extract query from input
            query = self._extract_query(input_data)
            
            # Use hybrid retrieval to combine multiple strategies
            retrieval_result = self.hybrid_retriever.retrieve(
                query=query,
                max_documents=self.max_documents
            )
            
            result = AgentResult(
                success=True,
                data=retrieval_result,
                metadata={
                    "num_documents": len(retrieval_result.documents),
                    "retrieval_strategies": list(self.retrievers.keys()),
                    "max_documents": self.max_documents
                }
            )
            
            self.log_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Error in retriever agent: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data=None,
                metadata={},
                error=error_msg
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for retrieval.
        
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
    
    def _extract_query(self, input_data: Any) -> str:
        """Extract query string from input data."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data["query"]
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance."""
        stats = {}
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'get_stats'):
                stats[name] = retriever.get_stats()
        return stats
