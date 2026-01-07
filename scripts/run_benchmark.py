#!/usr/bin/env python3
"""
Benchmark runner script for Multi-Agent RAG system.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any

from src.pipeline.ma_rag_pipeline import MARAGPipeline
from src.evaluation.benchmark import Benchmark
from src.utils.logger import setup_logger

logger = setup_logger("run_benchmark")


def load_test_data(data_file: str) -> List[Dict[str, Any]]:
    """Load test data from file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Convert to expected format
    test_data = []
    for item in data:
        test_data.append({
            "query": item.get("query", item.get("text", "")),
            "relevant_docs": item.get("relevant_docs", []),
            "answer": item.get("answer", "")
        })
    
    return test_data


def run_benchmark(config_path: str, test_data_file: str, output_path: str, 
                 dataset_name: str = "test") -> None:
    """Run benchmark on test data."""
    logger.info(f"Running benchmark on {dataset_name} dataset...")
    
    # Load test data
    test_data = load_test_data(test_data_file)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Initialize pipeline
    pipeline = MARAGPipeline(config_path)
    
    # Load documents into pipeline
    logger.info("Loading documents...")
    pipeline.load_documents("data/processed/ms_marco_documents.json")
    logger.info("Documents loaded successfully")
    
    # Initialize benchmark
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    benchmark = Benchmark(config)
    
    # Run benchmark
    result = benchmark.run_benchmark(pipeline, test_data, dataset_name)
    
    # Save results
    benchmark.save_results(result, output_path)
    
    # Print summary
    print(f"\nBenchmark Results for {dataset_name}:")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Retrieval MRR: {result.retrieval_metrics.mean_reciprocal_rank:.4f}")
    print(f"Ranking MAP: {result.ranking_metrics.map_score:.4f}")
    print(f"Generation ROUGE-L: {result.generation_metrics.rouge_l:.4f}")
    print(f"Evidence Faithfulness: {result.generation_metrics.evidence_faithfulness:.4f}")
    
    logger.info("Benchmark completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run benchmark for Multi-Agent RAG system")
    parser.add_argument("--config", default="config/agent_config.yaml", help="Configuration file")
    parser.add_argument("--test_data", required=True, help="Test data file")
    parser.add_argument("--output", required=True, help="Output results file")
    parser.add_argument("--dataset_name", default="test", help="Dataset name")
    
    args = parser.parse_args()
    
    # Check if test data file exists
    if not Path(args.test_data).exists():
        logger.error(f"Test data file not found: {args.test_data}")
        return
    
    # Run benchmark
    run_benchmark(args.config, args.test_data, args.output, args.dataset_name)


if __name__ == "__main__":
    main()
