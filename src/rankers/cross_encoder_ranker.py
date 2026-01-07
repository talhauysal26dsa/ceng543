"""
Cross-encoder ranker implementation for document re-ranking.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import torch
import numpy as np
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """Result from document ranking."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class CrossEncoderRanker:
    """
    Cross-encoder ranker for re-ranking retrieved documents.
    
    Uses cross-encoder models to score query-document pairs and
    re-rank documents based on relevance scores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cross-encoder ranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.batch_size = config.get("batch_size", 16)
        self.max_length = config.get("max_length", 512)
        
        # Initialize model
        self.model = CrossEncoder(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def rank(self, query: str, documents: List[Dict[str, Any]]) -> RankingResult:
        """
        Re-rank documents based on query relevance.
        
        Args:
            query: Query string
            documents: List of documents to rank
            
        Returns:
            RankingResult: Re-ranked documents and scores
        """
        if not documents:
            return RankingResult(
                documents=[],
                scores=[],
                metadata={"model_name": self.model_name, "num_documents": 0}
            )
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get("text", "")
            pairs.append([query, text])
        
        # Score pairs in batches
        scores = self._score_pairs(pairs)
        
        # Sort documents by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Separate documents and scores
        ranked_docs = [doc for doc, _ in doc_score_pairs]
        ranked_scores = [score for _, score in doc_score_pairs]
        
        return RankingResult(
            documents=ranked_docs,
            scores=ranked_scores,
            metadata={
                "model_name": self.model_name,
                "num_documents": len(ranked_docs),
                "batch_size": self.batch_size
            }
        )
    
    def _score_pairs(self, pairs: List[List[str]]) -> List[float]:
        """
        Score query-document pairs using the cross-encoder model.
        
        Args:
            pairs: List of [query, document] pairs
            
        Returns:
            List[float]: Relevance scores
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            try:
                batch_scores = self.model.predict(
                    batch_pairs,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Convert to list of floats
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.cpu().numpy()
                
                scores.extend(batch_scores.tolist())
                
            except Exception as e:
                logger.warning(f"Error scoring batch {i//self.batch_size}: {e}")
                # Add zero scores for failed batch
                scores.extend([0.0] * len(batch_pairs))
        
        return scores
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ranker."""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": str(self.device)
        }
