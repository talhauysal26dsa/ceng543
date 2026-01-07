"""
Ranker agent for Multi-Agent RAG system.
Performs cross-encoder re-ranking of retrieved documents.
"""

from typing import Any, Dict, List, Optional
import logging
import torch
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult
from ..rankers.cross_encoder_ranker import CrossEncoderRanker

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """Result from document ranking."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class RankerAgent(BaseAgent):
    """
    Agent responsible for re-ranking retrieved documents.
    
    Uses cross-encoder models to score query-document pairs and
    re-rank documents based on relevance scores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ranker agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("ranker_agent", config)
        
        # Check if ranking should be skipped
        self.skip_ranking = config.get("skip_ranking", False)
        
        # Initialize cross-encoder ranker only if needed
        model_config = config.get("model_config", {})
        self.max_documents = config.get("max_documents", 20)
        
        if not self.skip_ranking:
            self.ranker = CrossEncoderRanker(model_config)
        else:
            self.ranker = None
            logger.info("Ranker DISABLED - skip_ranking=True")
        
    def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Re-rank retrieved documents based on query relevance.
        
        Args:
            input_data: Dictionary containing 'query' and 'documents'
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Re-ranked documents and scores
        """
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="Invalid input data"
                )
            
            # Extract query and documents
            query = input_data["query"]
            documents = input_data["documents"]
            
            # If skip_ranking is True, pass documents through without ranking
            if self.skip_ranking:
                self.logger.info("Skipping ranking - passing documents through")
                ranking_result = RankingResult(
                    documents=documents[:self.max_documents],
                    scores=[1.0] * min(len(documents), self.max_documents),
                    metadata={"skipped": True}
                )
            else:
                # Limit number of documents to rank
                if len(documents) > self.max_documents:
                    documents = documents[:self.max_documents]
                    self.logger.info(f"Limited documents to {self.max_documents} for ranking")
                
                # Re-rank documents
                ranking_result = self.ranker.rank(
                    query=query,
                    documents=documents
                )
            
            result = AgentResult(
                success=True,
                data=ranking_result,
                metadata={
                    "num_documents": len(ranking_result.documents),
                    "max_documents": self.max_documents,
                    "model_name": self.ranker.model_name if self.ranker else "skipped",
                    "skip_ranking": self.skip_ranking
                }
            )
            
            self.log_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Error in ranker agent: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data=None,
                metadata={},
                error=error_msg
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for ranking.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid
        """
        if not isinstance(input_data, dict):
            return False
        
        if "query" not in input_data or "documents" not in input_data:
            return False
        
        query = input_data["query"]
        documents = input_data["documents"]
        
        if not isinstance(query, str) or len(query.strip()) == 0:
            return False
        
        if not isinstance(documents, list) or len(documents) == 0:
            return False
        
        # Check that documents have required fields
        for doc in documents:
            if not isinstance(doc, dict) or "text" not in doc:
                return False
        
        return True
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get statistics about ranking performance."""
        return {
            "model_name": self.ranker.model_name,
            "max_documents": self.max_documents,
            "device": str(self.ranker.device) if hasattr(self.ranker, 'device') else "unknown"
        }
