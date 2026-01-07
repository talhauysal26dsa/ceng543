#!/usr/bin/env python3
"""
Results analysis script for Multi-Agent RAG system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

from evaluation.benchmark import Benchmark
from utils.logger import setup_logger

logger = setup_logger("analyze_results")


def load_results(results_file: str) -> Dict[str, Any]:
    """Load benchmark results from file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_retrieval_performance(results: Dict[str, Any]) -> None:
    """Analyze retrieval performance metrics."""
    logger.info("Analyzing retrieval performance...")
    
    retrieval_metrics = results["retrieval_metrics"]
    
    print("\nRetrieval Performance:")
    print(f"MRR: {retrieval_metrics['mean_reciprocal_rank']:.4f}")
    
    print("\nRecall@K:")
    for k in [1, 5, 10, 20, 50, 100]:
        if f"recall_at_{k}" in retrieval_metrics:
            print(f"  Recall@{k}: {retrieval_metrics[f'recall_at_{k}']:.4f}")
    
    print("\nPrecision@K:")
    for k in [1, 5, 10, 20, 50, 100]:
        if f"precision_at_{k}" in retrieval_metrics:
            print(f"  Precision@{k}: {retrieval_metrics[f'precision_at_{k}']:.4f}")


def analyze_ranking_performance(results: Dict[str, Any]) -> None:
    """Analyze ranking performance metrics."""
    logger.info("Analyzing ranking performance...")
    
    ranking_metrics = results["ranking_metrics"]
    
    print("\nRanking Performance:")
    print(f"MAP: {ranking_metrics['map_score']:.4f}")
    print(f"MRR: {ranking_metrics['mrr']:.4f}")
    
    print("\nnDCG@K:")
    for k in [1, 5, 10, 20, 50]:
        if f"ndcg_at_{k}" in ranking_metrics:
            print(f"  nDCG@{k}: {ranking_metrics[f'ndcg_at_{k}']:.4f}")


def analyze_generation_performance(results: Dict[str, Any]) -> None:
    """Analyze generation performance metrics."""
    logger.info("Analyzing generation performance...")
    
    generation_metrics = results["generation_metrics"]
    
    print("\nGeneration Performance:")
    print(f"ROUGE-L: {generation_metrics['rouge_l']:.4f}")
    print(f"BLEU Score: {generation_metrics['bleu_score']:.4f}")
    print(f"Evidence Faithfulness: {generation_metrics['evidence_faithfulness']:.4f}")
    print(f"Average Answer Length: {generation_metrics['answer_length']} words")


def create_performance_plots(results: Dict[str, Any], output_dir: str) -> None:
    """Create performance visualization plots."""
    logger.info("Creating performance plots...")
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot retrieval metrics
    retrieval_metrics = results["retrieval_metrics"]
    
    # Recall@K plot
    recall_data = []
    for k in [1, 5, 10, 20, 50, 100]:
        if f"recall_at_{k}" in retrieval_metrics:
            recall_data.append({
                "K": k,
                "Recall": retrieval_metrics[f"recall_at_{k}"]
            })
    
    if recall_data:
        recall_df = pd.DataFrame(recall_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall_df["K"], recall_df["Recall"], marker='o')
        plt.title("Recall@K Performance")
        plt.xlabel("K")
        plt.ylabel("Recall")
        plt.grid(True)
        plt.savefig(Path(output_dir) / "recall_at_k.png")
        plt.close()
    
    # Precision@K plot
    precision_data = []
    for k in [1, 5, 10, 20, 50, 100]:
        if f"precision_at_{k}" in retrieval_metrics:
            precision_data.append({
                "K": k,
                "Precision": retrieval_metrics[f"precision_at_{k}"]
            })
    
    if precision_data:
        precision_df = pd.DataFrame(precision_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(precision_df["K"], precision_df["Precision"], marker='o')
        plt.title("Precision@K Performance")
        plt.xlabel("K")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.savefig(Path(output_dir) / "precision_at_k.png")
        plt.close()
    
    # Ranking metrics plot
    ranking_metrics = results["ranking_metrics"]
    
    ranking_data = []
    for k in [1, 5, 10, 20, 50]:
        if f"ndcg_at_{k}" in ranking_metrics:
            ranking_data.append({
                "K": k,
                "nDCG": ranking_metrics[f"ndcg_at_{k}"]
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        
        plt.figure(figsize=(10, 6))
        plt.plot(ranking_df["K"], ranking_df["nDCG"], marker='o')
        plt.title("nDCG@K Performance")
        plt.xlabel("K")
        plt.ylabel("nDCG")
        plt.grid(True)
        plt.savefig(Path(output_dir) / "ndcg_at_k.png")
        plt.close()
    
    logger.info(f"Performance plots saved to {output_dir}")


def compare_results(results_files: List[str], output_file: str) -> None:
    """Compare results from multiple experiments."""
    logger.info("Comparing results from multiple experiments...")
    
    comparison_data = []
    
    for results_file in results_files:
        results = load_results(results_file)
        
        # Extract key metrics
        comparison_data.append({
            "experiment": Path(results_file).stem,
            "mrr": results["retrieval_metrics"]["mean_reciprocal_rank"],
            "map": results["ranking_metrics"]["map_score"],
            "rouge_l": results["generation_metrics"]["rouge_l"],
            "evidence_faithfulness": results["generation_metrics"]["evidence_faithfulness"],
            "execution_time": results["execution_time"]
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_df.to_csv(output_file, index=False)
    
    # Print comparison
    print("\nExperiment Comparison:")
    print(comparison_df.to_string(index=False))
    
    logger.info(f"Comparison saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results for Multi-Agent RAG system")
    parser.add_argument("--results", help="Results file to analyze")
    parser.add_argument("--compare", nargs="+", help="Multiple results files to compare")
    parser.add_argument("--output_dir", default="experiments/results/analysis", help="Output directory for plots")
    parser.add_argument("--comparison_output", default="experiments/results/comparison.csv", help="Output file for comparison")
    
    args = parser.parse_args()
    
    if args.results:
        # Analyze single results file
        results = load_results(args.results)
        
        analyze_retrieval_performance(results)
        analyze_ranking_performance(results)
        analyze_generation_performance(results)
        
        create_performance_plots(results, args.output_dir)
    
    if args.compare:
        # Compare multiple results files
        compare_results(args.compare, args.comparison_output)
    
    logger.info("Results analysis completed!")


if __name__ == "__main__":
    main()
