"""
Ranking metrics implementation for Multi-Agent RAG system.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankingMetricsResult:
    """Result from ranking metrics calculation."""
    map_score: float
    ndcg_at_k: Dict[int, float]
    mrr: float
    metadata: Dict[str, Any]


class RankingMetrics:
    """
    Metrics for evaluating ranking performance.
    
    Implements standard ranking metrics including:
    - Mean Average Precision (MAP)
    - Normalized Discounted Cumulative Gain (nDCG@K)
    - Mean Reciprocal Rank (MRR)
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize ranking metrics.
        
        Args:
            k_values: List of K values for nDCG@K
        """
        self.k_values = k_values or [1, 5, 10, 20, 50]
        
    def calculate_metrics(self, ranked_docs: List[List[str]], 
                         relevance_scores: List[List[float]]) -> RankingMetricsResult:
        """
        Calculate ranking metrics.
        
        Args:
            ranked_docs: List of ranked document IDs for each query
            relevance_scores: List of relevance scores for each query
            
        Returns:
            RankingMetricsResult: Calculated metrics
        """
        if len(ranked_docs) != len(relevance_scores):
            raise ValueError("Number of queries must match between ranked docs and relevance scores")
        
        num_queries = len(ranked_docs)
        
        # Calculate MAP
        map_score = self._calculate_map(ranked_docs, relevance_scores)
        
        # Calculate nDCG@K for each K value
        ndcg_at_k = {}
        for k in self.k_values:
            ndcg_scores = []
            for i in range(num_queries):
                ndcg = self._calculate_ndcg_at_k(
                    ranked_docs[i][:k], 
                    relevance_scores[i], 
                    k
                )
                ndcg_scores.append(ndcg)
            ndcg_at_k[k] = np.mean(ndcg_scores)
        
        # Calculate MRR
        mrr = self._calculate_mrr(ranked_docs, relevance_scores)
        
        return RankingMetricsResult(
            map_score=map_score,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            metadata={
                "num_queries": num_queries,
                "k_values": self.k_values
            }
        )
    
    def _calculate_map(self, ranked_docs: List[List[str]], 
                      relevance_scores: List[Dict[str, float]]) -> float:
        """
        Calculate Mean Average Precision.
        
        Args:
            ranked_docs: List of ranked document IDs for each query
            relevance_scores: List of relevance score dicts for each query
            
        Returns:
            float: Mean Average Precision
        """
        average_precisions = []
        
        for i in range(len(ranked_docs)):
            docs = ranked_docs[i]
            relevance_map = relevance_scores[i]  # Already a dict
            
            # Calculate average precision
            relevant_docs = [doc for doc, score in relevance_map.items() if score > 0]
            
            if not relevant_docs:
                average_precisions.append(0.0)
                continue
            
            precision_sum = 0.0
            relevant_count = 0
            
            for rank, doc in enumerate(docs, 1):
                if doc in relevant_docs:
                    relevant_count += 1
                    precision = relevant_count / rank
                    precision_sum += precision
            
            if relevant_count > 0:
                average_precisions.append(precision_sum / relevant_count)
            else:
                average_precisions.append(0.0)
        
        return np.mean(average_precisions)
    
    def _calculate_ndcg_at_k(self, ranked_docs: List[str], 
                           relevance_scores: List[float], k: int) -> float:
        """
        Calculate nDCG@K for a single query.
        
        Args:
            ranked_docs: Ranked document IDs
            relevance_scores: Relevance scores
            k: Number of top documents to consider
            
        Returns:
            float: nDCG@K score
        """
        if k == 0:
            return 0.0
        
        # Calculate DCG@K
        dcg = 0.0
        for i in range(min(k, len(ranked_docs))):
            doc = ranked_docs[i]
            if doc in relevance_scores:
                score = relevance_scores[doc]
                dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@K (ideal DCG)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_scores))):
            idcg += ideal_scores[i] / np.log2(i + 2)
        
        # Calculate nDCG@K
        if idcg > 0:
            return dcg / idcg
        else:
            return 0.0
    
    def _calculate_mrr(self, ranked_docs: List[List[str]], 
                      relevance_scores: List[Dict[str, float]]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            ranked_docs: List of ranked document IDs for each query
            relevance_scores: List of relevance score dicts for each query
            
        Returns:
            float: Mean Reciprocal Rank
        """
        reciprocal_ranks = []
        
        for i in range(len(ranked_docs)):
            docs = ranked_docs[i]
            relevance_map = relevance_scores[i]  # Already a dict
            
            # Find first relevant document (score > 0)
            for rank, doc in enumerate(docs, 1):
                if doc in relevance_map and relevance_map[doc] > 0:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def calculate_single_query_metrics(self, ranked_docs: List[str], 
                                     relevance_scores: Dict[str, float], 
                                     k: int = 10) -> Dict[str, float]:
        """
        Calculate metrics for a single query.
        
        Args:
            ranked_docs: Ranked document IDs
            relevance_scores: Relevance scores for documents
            k: Number of top documents to consider
            
        Returns:
            Dict[str, float]: Metrics for the query
        """
        # Calculate MAP
        relevant_docs = [doc for doc, score in relevance_scores.items() if score > 0]
        
        if not relevant_docs:
            return {"map": 0.0, "ndcg": 0.0, "mrr": 0.0}
        
        precision_sum = 0.0
        relevant_count = 0
        
        for rank, doc in enumerate(ranked_docs[:k], 1):
            if doc in relevant_docs:
                relevant_count += 1
                precision = relevant_count / rank
                precision_sum += precision
        
        map_score = precision_sum / relevant_count if relevant_count > 0 else 0.0
        
        # Calculate nDCG@K
        ndcg = self._calculate_ndcg_at_k(ranked_docs, relevance_scores, k)
        
        # Calculate MRR
        mrr = 0.0
        for rank, doc in enumerate(ranked_docs, 1):
            if doc in relevance_scores and relevance_scores[doc] > 0:
                mrr = 1.0 / rank
                break
        
        return {
            "map": map_score,
            "ndcg": ndcg,
            "mrr": mrr
        }
