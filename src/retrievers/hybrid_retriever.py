"""
Hybrid retriever combining BM25 and dense retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass

from .bm25_retriever import BM25Retriever, BM25Result
from .dense_retriever import DenseRetriever, DenseResult

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense retrieval methods.
    
    Combines sparse (BM25) and dense (sentence transformer) retrieval
    using reciprocal rank fusion or weighted combination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.retrievers = config.get("retrievers", {})
        self.max_documents = config.get("max_documents", 100)
        self.fusion_method = config.get("fusion_method", "rrf")  # rrf or weighted
        self.bm25_weight = config.get("bm25_weight", 0.5)
        self.dense_weight = config.get("dense_weight", 0.5)
        self.rrf_k = config.get("rrf_k", 60)
        
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for all retrievers.
        
        Args:
            documents: List of documents to index
        """
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'index_documents'):
                retriever.index_documents(documents)
                logger.info(f"Indexed documents for {name} retriever")
    
    def retrieve(self, query: str, max_documents: int = None) -> HybridResult:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Query string
            max_documents: Maximum number of documents to return
            
        Returns:
            HybridResult: Retrieved documents and scores
        """
        if max_documents is None:
            max_documents = self.max_documents
        
        # Get results from all retrievers
        retrieval_results = {}
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'retrieve'):
                try:
                    result = retriever.retrieve(query, max_documents)
                    retrieval_results[name] = result
                except Exception as e:
                    logger.warning(f"Retriever {name} failed: {e}")
        
        if not retrieval_results:
            raise ValueError("No retrievers available")
        
        # Combine results
        if self.fusion_method == "rrf":
            combined_result = self._reciprocal_rank_fusion(retrieval_results, max_documents)
        elif self.fusion_method == "weighted":
            combined_result = self._weighted_combination(retrieval_results, max_documents)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return combined_result
    
    def _reciprocal_rank_fusion(self, results: Dict[str, Any], max_documents: int) -> HybridResult:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            results: Results from different retrievers
            max_documents: Maximum number of documents to return
            
        Returns:
            HybridResult: Combined results
        """
        doc_scores = {}
        
        for retriever_name, result in results.items():
            for rank, (doc, score) in enumerate(zip(result.documents, result.scores)):
                doc_id = doc.get("id", id(doc))
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                doc_scores[doc_id]["score"] += rrf_score
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:max_documents]
        
        documents = [item["doc"] for item in sorted_docs]
        scores = [item["score"] for item in sorted_docs]
        
        return HybridResult(
            documents=documents,
            scores=scores,
            metadata={
                "fusion_method": "rrf",
                "rrf_k": self.rrf_k,
                "retrievers_used": list(results.keys()),
                "num_documents": len(documents)
            }
        )
    
    def _weighted_combination(self, results: Dict[str, Any], max_documents: int) -> HybridResult:
        """
        Combine results using weighted combination.
        
        Args:
            results: Results from different retrievers
            max_documents: Maximum number of documents to return
            
        Returns:
            HybridResult: Combined results
        """
        doc_scores = {}
        
        for retriever_name, result in results.items():
            weight = self._get_retriever_weight(retriever_name)
            
            for doc, score in zip(result.documents, result.scores):
                doc_id = doc.get("id", id(doc))
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                
                doc_scores[doc_id]["score"] += weight * score
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:max_documents]
        
        documents = [item["doc"] for item in sorted_docs]
        scores = [item["score"] for item in sorted_docs]
        
        return HybridResult(
            documents=documents,
            scores=scores,
            metadata={
                "fusion_method": "weighted",
                "retrievers_used": list(results.keys()),
                "num_documents": len(documents)
            }
        )
    
    def _get_retriever_weight(self, retriever_name: str) -> float:
        """Get weight for a specific retriever."""
        if "bm25" in retriever_name.lower():
            return self.bm25_weight
        elif "dense" in retriever_name.lower():
            return self.dense_weight
        else:
            return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid retriever."""
        stats = {
            "fusion_method": self.fusion_method,
            "max_documents": self.max_documents,
            "retrievers": {}
        }
        
        for name, retriever in self.retrievers.items():
            if hasattr(retriever, 'get_stats'):
                stats["retrievers"][name] = retriever.get_stats()
        
        return stats
