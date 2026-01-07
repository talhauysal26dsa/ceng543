"""
Benchmark implementation for Multi-Agent RAG system.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .retrieval_metrics import RetrievalMetrics, RetrievalMetricsResult
from .ranking_metrics import RankingMetrics, RankingMetricsResult
from .generation_metrics import GenerationMetrics, GenerationMetricsResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from benchmark execution."""
    retrieval_metrics: RetrievalMetricsResult
    ranking_metrics: RankingMetricsResult
    generation_metrics: GenerationMetricsResult
    execution_time: float
    avg_latency_per_query: float
    total_cost: float
    avg_cost_per_query: float
    metadata: Dict[str, Any]


class Benchmark:
    """
    Benchmark runner for Multi-Agent RAG system.
    
    Evaluates the complete pipeline on multiple datasets and
    provides comprehensive performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.retrieval_metrics = RetrievalMetrics()
        self.ranking_metrics = RankingMetrics()
        self.generation_metrics = GenerationMetrics()
        
    def run_benchmark(self, pipeline, test_data: List[Dict[str, Any]], 
                     dataset_name: str = "test") -> BenchmarkResult:
        """
        Run benchmark on test data.
        
        Args:
            pipeline: RAG pipeline to evaluate
            test_data: Test data containing queries and ground truth
            dataset_name: Name of the dataset
            
        Returns:
            BenchmarkResult: Benchmark results
        """
        start_time = time.time()
        
        logger.info(f"Running benchmark on {dataset_name} dataset")
        logger.info(f"Test data size: {len(test_data)}")
        
        # Prepare data for evaluation
        queries = [item["query"] for item in test_data]
        
        # Extract ground truth - support both formats
        ground_truth = []
        for item in test_data:
            # Try 'relevant_docs' first (list format)
            rel_docs = item.get("relevant_docs")
            if rel_docs:
                ground_truth.append(rel_docs)
            else:
                # Try 'qrels' (dict format) - extract keys
                qrels = item.get("qrels", {})
                if isinstance(qrels, dict):
                    ground_truth.append(list(qrels.keys()))
                else:
                    ground_truth.append([])
        
        reference_answers = [item.get("answer", "") for item in test_data]
        
        # Run pipeline on test data with latency tracking
        results = []
        query_latencies = []
        query_costs = []
        
        total_queries = len(queries)
        start_time_total = time.time()
        
        for i, query in enumerate(queries):
            # Progress bar
            progress = (i + 1) / total_queries
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '=' * filled + '-' * (bar_length - filled)
            
            # Time estimate
            if i > 0:
                elapsed = time.time() - start_time_total
                avg_per_query = elapsed / i
                remaining = avg_per_query * (total_queries - i)
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"ETA: {eta_min}m {eta_sec}s"
            else:
                eta_str = "ETA: calculating..."
            
            print(f"\r[{bar}] {i+1}/{total_queries} ({progress*100:.1f}%) - {eta_str}     ", end='', flush=True)
            logger.info(f"Processing query {i+1}/{len(queries)}")
            
            try:
                # Track query latency
                query_start = time.time()
                
                # Run pipeline
                result = pipeline.process(query)
                
                query_latency = time.time() - query_start
                query_latencies.append(query_latency)
                
                # Estimate cost (tokens * cost_per_token)
                # Assuming average cost based on query/answer length
                query_cost = self._estimate_query_cost(query, result)
                query_costs.append(query_cost)
                
                if result.success:
                    # Extract doc IDs from evidence (not documents - that doesn't exist)
                    retrieved_doc_ids = []
                    if hasattr(result.data, 'evidence') and result.data.evidence:
                        retrieved_doc_ids = [e.get('source', e.get('doc_id', '')) for e in result.data.evidence]
                    
                    results.append({
                        "query": query,
                        "retrieved_docs": retrieved_doc_ids,  # Use evidence source IDs
                        "generated_answer": result.data.answer if hasattr(result.data, 'answer') else "",
                        "evidence": result.data.evidence if hasattr(result.data, 'evidence') else [],
                        "latency": query_latency,
                        "cost": query_cost
                    })
                else:
                    logger.warning(f"Pipeline failed for query {i+1}: {result.error}")
                    results.append({
                        "query": query,
                        "retrieved_docs": [],
                        "generated_answer": "",
                        "evidence": [],
                        "latency": query_latency,
                        "cost": 0.0
                    })
                    
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                query_latency = time.time() - query_start if 'query_start' in locals() else 0.0
                query_latencies.append(query_latency)
                query_costs.append(0.0)
                results.append({
                    "query": query,
                    "retrieved_docs": [],
                    "generated_answer": "",
                    "evidence": [],
                    "latency": query_latency,
                    "cost": 0.0
                })
        
        # Calculate metrics
        retrieval_metrics = self._calculate_retrieval_metrics(results, ground_truth)
        ranking_metrics = self._calculate_ranking_metrics(results, ground_truth)
        generation_metrics = self._calculate_generation_metrics(results, reference_answers)
        
        execution_time = time.time() - start_time
        avg_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
        total_cost = sum(query_costs)
        avg_cost = total_cost / len(query_costs) if query_costs else 0.0
        
        return BenchmarkResult(
            retrieval_metrics=retrieval_metrics,
            ranking_metrics=ranking_metrics,
            generation_metrics=generation_metrics,
            execution_time=execution_time,
            avg_latency_per_query=avg_latency,
            total_cost=total_cost,
            avg_cost_per_query=avg_cost,
            metadata={
                "dataset_name": dataset_name,
                "num_queries": len(queries),
                "successful_queries": len([r for r in results if r["retrieved_docs"]]),
                "pipeline_config": self.config,
                "query_latencies": query_latencies,
                "query_costs": query_costs
            }
        )
    
    def _calculate_retrieval_metrics(self, results: List[Dict[str, Any]], 
                                   ground_truth: List[List[str]]) -> RetrievalMetricsResult:
        """Calculate retrieval metrics."""
        retrieved_docs = [result["retrieved_docs"] for result in results]
        
        # Convert to document IDs if needed
        retrieved_ids = []
        for docs in retrieved_docs:
            if docs and isinstance(docs[0], dict):
                doc_ids = [doc.get("id") or doc.get("doc_id") or doc.get("title") or str(i) for i, doc in enumerate(docs)]
            else:
                doc_ids = [str(doc) for doc in docs]
            retrieved_ids.append(doc_ids)
        
        return self.retrieval_metrics.calculate_metrics(retrieved_ids, ground_truth)
    
    def _calculate_ranking_metrics(self, results: List[Dict[str, Any]], 
                                 ground_truth: List[List[str]]) -> RankingMetricsResult:
        """Calculate ranking metrics."""
        # For ranking metrics, we need relevance scores
        # This is a simplified version - in practice, you'd need ground truth relevance scores
        ranked_docs = [result["retrieved_docs"] for result in results]
        
        # Convert to document IDs and create dummy relevance scores
        ranked_ids = []
        relevance_scores = []
        
        for i, docs in enumerate(ranked_docs):
            if docs and isinstance(docs[0], dict):
                doc_ids = [doc.get("id") or doc.get("doc_id") or doc.get("title") or str(j) for j, doc in enumerate(docs)]
            else:
                doc_ids = [str(doc) for doc in docs]
            
            ranked_ids.append(doc_ids)
            
            # Create dummy relevance scores (1 for relevant docs, 0 for others)
            gt_docs = set(ground_truth[i])
            scores = {doc_id: 1.0 if doc_id in gt_docs else 0.0 for doc_id in doc_ids}
            relevance_scores.append(scores)
        
        return self.ranking_metrics.calculate_metrics(ranked_ids, relevance_scores)
    
    def _calculate_generation_metrics(self, results: List[Dict[str, Any]], 
                                    reference_answers: List[str]) -> GenerationMetricsResult:
        """Calculate generation metrics."""
        generated_answers = [result["generated_answer"] for result in results]
        evidence_docs = [result["evidence"] for result in results]
        
        return self.generation_metrics.calculate_metrics(
            generated_answers, reference_answers, evidence_docs
        )
    
    def _estimate_query_cost(self, query: str, result: Any) -> float:
        """
        Estimate cost per query based on token usage.
        
        Args:
            query: Input query
            result: Pipeline result
            
        Returns:
            float: Estimated cost in USD
        """
        # Rough estimation based on GPT-3.5-turbo pricing
        # Input: $0.0015 per 1K tokens, Output: $0.002 per 1K tokens
        INPUT_COST_PER_1K = 0.0015
        OUTPUT_COST_PER_1K = 0.002
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        query_tokens = len(query) / 4
        
        # Estimate output tokens
        output_tokens = 0
        if hasattr(result, 'data'):
            if hasattr(result.data, 'answer'):
                output_tokens = len(result.data.answer) / 4
            elif hasattr(result.data, 'documents'):
                # Estimate based on retrieved documents
                output_tokens = len(str(result.data.documents)[:500]) / 4
        
        # Calculate cost
        input_cost = (query_tokens / 1000) * INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
        total_cost = input_cost + output_cost
        
        return total_cost
    
    def save_results(self, result: BenchmarkResult, output_path: str) -> None:
        """
        Save benchmark results to file.
        
        Args:
            result: Benchmark result to save
            output_path: Path to save results
        """
        output_data = {
            "retrieval_metrics": {
                "recall_at_k": result.retrieval_metrics.recall_at_k,
                "precision_at_k": result.retrieval_metrics.precision_at_k,
                "f1_at_k": result.retrieval_metrics.f1_at_k,
                "mean_reciprocal_rank": result.retrieval_metrics.mean_reciprocal_rank
            },
            "ranking_metrics": {
                "map_score": result.ranking_metrics.map_score,
                "ndcg_at_k": result.ranking_metrics.ndcg_at_k,
                "mrr": result.ranking_metrics.mrr
            },
            "generation_metrics": {
                "rouge_l": result.generation_metrics.rouge_l,
                "bleu_score": result.generation_metrics.bleu_score,
                "evidence_faithfulness": result.generation_metrics.evidence_faithfulness,
                "citation_quality": result.generation_metrics.citation_quality,
                "answer_length": result.generation_metrics.answer_length
            },
            "performance_metrics": {
                "execution_time": result.execution_time,
                "avg_latency_per_query": result.avg_latency_per_query,
                "total_cost": result.total_cost,
                "avg_cost_per_query": result.avg_cost_per_query
            },
            "metadata": result.metadata
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Avg Latency: {result.avg_latency_per_query:.2f}s, Avg Cost: ${result.avg_cost_per_query:.4f}")
    
    def load_results(self, input_path: str) -> BenchmarkResult:
        """
        Load benchmark results from file.
        
        Args:
            input_path: Path to load results from
            
        Returns:
            BenchmarkResult: Loaded benchmark result
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        return BenchmarkResult(
            retrieval_metrics=RetrievalMetricsResult(**data["retrieval_metrics"]),
            ranking_metrics=RankingMetricsResult(**data["ranking_metrics"]),
            generation_metrics=GenerationMetricsResult(**data["generation_metrics"]),
            execution_time=data["execution_time"],
            metadata=data["metadata"]
        )
