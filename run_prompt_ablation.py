"""
Prompt Technique Ablation Study
Tests Chain of Thought (CoT) and Few-Shot Learning prompting techniques
across all three datasets: MS MARCO, HotpotQA, FEVER.

Configuration:
- Baseline: Simple prompt (original)
- CoT: Chain of Thought step-by-step reasoning
- Few-Shot: 3-shot in-context learning
"""

import json
import yaml
import os
from datetime import datetime
from src.evaluation.benchmark import Benchmark
from src.pipeline.ma_rag_pipeline import MARAGPipeline

print("=" * 80)
print("PROMPT TECHNIQUE ABLATION STUDY")
print("Testing: Baseline vs Chain of Thought vs Few-Shot Learning")
print("=" * 80)

# Datasets and their configurations
DATASETS = {
    'msmarco': {
        'test_data': 'data/test/ms_marco_test_100.json',
        'documents': 'data/processed/ms_marco_documents.json',
        'name': 'MS MARCO'
    },
    'hotpot': {
        'test_data': 'data/test/hotpotqa_test_100.json',
        'documents': 'data/processed/hotpotqa_documents.json',
        'name': 'HotpotQA'
    },
    'fever': {
        'test_data': 'data/test/fever_test_100.json',
        'documents': 'data/processed/fever_test_docs.json',
        'name': 'FEVER'
    }
}

# Prompt techniques and their configs
PROMPT_TECHNIQUES = {
    'baseline': 'config/agent_config.yaml',
    'cot': 'config/agent_config_cot.yaml',
    'fewshot': 'config/agent_config_fewshot.yaml'
}

def run_single_ablation(dataset_key, prompt_key, dataset_info, config_path):
    """Run a single ablation experiment."""
    output_file = f'experiments/results/prompt_ablation_{dataset_key}_{prompt_key}.json'
    
    # Skip if already exists
    if os.path.exists(output_file):
        print(f"\n[SKIP] {dataset_info['name']}/{prompt_key} - Result already exists: {output_file}")
        # Load existing result for comparison
        try:
            with open(output_file, 'r') as f:
                existing = json.load(f)
            return existing  # Return dict instead of BenchmarkResult object
        except:
            pass
        return None
    
    print(f"\n{'='*60}")
    print(f"[{dataset_info['name']}] - {prompt_key.upper()}")
    print(f"Config: {config_path}")
    print("=" * 60)
    
    # Check if test data exists
    if not os.path.exists(dataset_info['test_data']):
        print(f"[SKIP] Test data not found: {dataset_info['test_data']}")
        return None
    
    if not os.path.exists(dataset_info['documents']):
        print(f"[SKIP] Documents not found: {dataset_info['documents']}")
        return None
    
    # Load test data
    print(f"[*] Loading test data from {dataset_info['test_data']}...")
    with open(dataset_info['test_data'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"[OK] Loaded {len(test_data)} test queries")
    
    # Initialize pipeline
    print(f"[*] Initializing pipeline...")
    pipeline = MARAGPipeline(config_path)
    
    # Load documents
    print(f"[*] Loading documents from {dataset_info['documents']}...")
    pipeline.load_documents(dataset_info['documents'])
    print("[OK] Documents loaded!")
    
    # Load config for benchmark
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run benchmark
    benchmark = Benchmark(config)
    print("[*] Running benchmark...")
    result = benchmark.run_benchmark(pipeline, test_data, f'prompt_{dataset_key}_{prompt_key}')
    
    # Save results
    output_file = f'experiments/results/prompt_ablation_{dataset_key}_{prompt_key}.json'
    benchmark.save_results(result, output_file)
    print(f"[OK] Results saved to {output_file}")
    
    return result


def get_ndcg(result):
    """Extract nDCG@10 from result."""
    if result and hasattr(result, 'ranking_metrics') and hasattr(result.ranking_metrics, 'ndcg_at_k'):
        return result.ranking_metrics.ndcg_at_k.get(10, 0.0)
    return 0.0


def calc_improvement(new_val, old_val):
    """Calculate percentage improvement."""
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / old_val * 100


def main():
    """Main function to run all ablations."""
    all_results = {}
    
    # Run all experiments
    for dataset_key, dataset_info in DATASETS.items():
        all_results[dataset_key] = {}
        
        for prompt_key, config_path in PROMPT_TECHNIQUES.items():
            try:
                result = run_single_ablation(dataset_key, prompt_key, dataset_info, config_path)
                all_results[dataset_key][prompt_key] = result
            except Exception as e:
                print(f"[ERROR] Failed {dataset_key}/{prompt_key}: {e}")
                all_results[dataset_key][prompt_key] = None
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("PROMPT TECHNIQUE ABLATION RESULTS")
    print("=" * 80)
    
    print(f"\n{'Dataset':<12} {'Technique':<12} {'MRR':>8} {'MAP':>8} {'nDCG@10':>10} {'Faith':>8} {'Improvement':>12}")
    print("-" * 80)
    
    for dataset_key, dataset_info in DATASETS.items():
        dataset_results = all_results.get(dataset_key, {})
        baseline_result = dataset_results.get('baseline')
        
        for prompt_key in PROMPT_TECHNIQUES.keys():
            result = dataset_results.get(prompt_key)
            
            if result is None:
                print(f"{dataset_info['name']:<12} {prompt_key:<12} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>12}")
                continue
            
            mrr = result.retrieval_metrics.mean_reciprocal_rank
            map_score = result.ranking_metrics.map_score
            ndcg = get_ndcg(result)
            faith = result.generation_metrics.evidence_faithfulness
            
            if prompt_key == 'baseline':
                improvement = "baseline"
            elif baseline_result:
                baseline_faith = baseline_result.generation_metrics.evidence_faithfulness
                impr = calc_improvement(faith, baseline_faith)
                improvement = f"{impr:+.1f}%"
            else:
                improvement = "N/A"
            
            print(f"{dataset_info['name']:<12} {prompt_key:<12} {mrr:>8.4f} {map_score:>8.4f} {ndcg:>10.4f} {faith:>8.4f} {improvement:>12}")
        
        print("-" * 80)
    
    # Summary of prompt technique effectiveness
    print("\n" + "=" * 80)
    print("PROMPT TECHNIQUE SUMMARY")
    print("=" * 80)
    
    for prompt_key in ['cot', 'fewshot']:
        improvements = []
        for dataset_key in DATASETS.keys():
            baseline = all_results.get(dataset_key, {}).get('baseline')
            technique = all_results.get(dataset_key, {}).get(prompt_key)
            
            if baseline and technique:
                baseline_faith = baseline.generation_metrics.evidence_faithfulness
                technique_faith = technique.generation_metrics.evidence_faithfulness
                impr = calc_improvement(technique_faith, baseline_faith)
                improvements.append(impr)
        
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            technique_name = "Chain of Thought" if prompt_key == 'cot' else "Few-Shot Learning"
            print(f"\n{technique_name}:")
            print(f"  Average Faithfulness Improvement: {avg_improvement:+.2f}%")
    
    print("\n" + "=" * 80)
    print("[OK] PROMPT TECHNIQUE ABLATION COMPLETED!")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
