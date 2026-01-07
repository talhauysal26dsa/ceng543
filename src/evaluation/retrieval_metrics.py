"""
Retrieval metrics implementation for Multi-Agent RAG system.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetricsResult:
    """Result from retrieval metrics calculation."""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    metadata: Dict[str, Any]


class RetrievalMetrics:
    """
    Metrics for evaluating retrieval performance.
    
    Implements standard retrieval metrics including:
    - Recall@K
    - Precision@K
    - F1@K
    - Mean Reciprocal Rank (MRR)
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize retrieval metrics.
        
        Args:
            k_values: List of K values for Recall@K, Precision@K, F1@K
        """
        self.k_values = k_values or [1, 5, 10, 20, 50, 100]
        
    def calculate_metrics(self, retrieved_docs: List[List[str]], 
                         relevant_docs: List[List[str]]) -> RetrievalMetricsResult:
        """
        Calculate retrieval metrics.
        
        Args:
            retrieved_docs: List of retrieved document IDs for each query
            relevant_docs: List of relevant document IDs for each query
            
        Returns:
            RetrievalMetricsResult: Calculated metrics
        """
        if len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Number of queries must match between retrieved and relevant docs")
        
        num_queries = len(retrieved_docs)
        
        # Calculate metrics for each K value
        recall_at_k = {}
        precision_at_k = {}
        f1_at_k = {}
        
        for k in self.k_values:
            recalls = []
            precisions = []
            f1_scores = []
            
            for i in range(num_queries):
                retrieved = set(retrieved_docs[i][:k])
                relevant = set(relevant_docs[i])
                
                # Calculate recall
                if len(relevant) > 0:
                    recall = len(retrieved.intersection(relevant)) / len(relevant)
                else:
                    recall = 0.0
                recalls.append(recall)
                
                # Calculate precision
                if len(retrieved) > 0:
                    precision = len(retrieved.intersection(relevant)) / len(retrieved)
                else:
                    precision = 0.0
                precisions.append(precision)
                
                # Calculate F1
                if recall + precision > 0:
                    f1 = 2 * (recall * precision) / (recall + precision)
                else:
                    f1 = 0.0
                f1_scores.append(f1)
            
            recall_at_k[k] = np.mean(recalls)
            precision_at_k[k] = np.mean(precisions)
            f1_at_k[k] = np.mean(f1_scores)
        
        # Calculate Mean Reciprocal Rank
        mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
        
        return RetrievalMetricsResult(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            f1_at_k=f1_at_k,
            mean_reciprocal_rank=mrr,
            metadata={
                "num_queries": num_queries,
                "k_values": self.k_values
            }
        )
    
    def _calculate_mrr(self, retrieved_docs: List[List[str]], 
                      relevant_docs: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_docs: List of retrieved document IDs for each query
            relevant_docs: List of relevant document IDs for each query
            
        Returns:
            float: Mean Reciprocal Rank
        """
        reciprocal_ranks = []
        
        for i in range(len(retrieved_docs)):
            retrieved = retrieved_docs[i]
            relevant = set(relevant_docs[i])
            
            if not relevant:
                reciprocal_ranks.append(0.0)
                continue
            
            # Find first relevant document
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def calculate_single_query_metrics(self, retrieved_docs: List[str], 
                                     relevant_docs: List[str], k: int = 10) -> Dict[str, float]:
        """
        Calculate metrics for a single query.
        
        Args:
            retrieved_docs: Retrieved document IDs
            relevant_docs: Relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Dict[str, float]: Metrics for the query
        """
        retrieved = set(retrieved_docs[:k])
        relevant = set(relevant_docs)
        
        # Calculate recall
        if len(relevant) > 0:
            recall = len(retrieved.intersection(relevant)) / len(relevant)
        else:
            recall = 0.0
        
        # Calculate precision
        if len(retrieved) > 0:
            precision = len(retrieved.intersection(relevant)) / len(retrieved)
        else:
            precision = 0.0
        
        # Calculate F1
        if recall + precision > 0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0.0
        
        # Calculate reciprocal rank
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        
        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "reciprocal_rank": reciprocal_rank
        }
