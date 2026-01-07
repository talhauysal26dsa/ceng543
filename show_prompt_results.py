"""Show prompt ablation results summary."""
import json
import os

files = [f for f in os.listdir('experiments/results') if f.startswith('prompt_ablation')]
print('PROMPT ABLATION RESULTS')
print('='*80)
print(f"{'Dataset':<12} {'Technique':<12} {'MRR':>8} {'MAP':>8} {'nDCG@10':>10} {'Faith':>8}")
print('-'*80)

for f in sorted(files):
    with open(f'experiments/results/{f}') as fp:
        data = json.load(fp)
    parts = f.replace('prompt_ablation_','').replace('.json','').split('_')
    dataset = parts[0].upper()
    tech = parts[1]
    mrr = data.get('retrieval_metrics',{}).get('mean_reciprocal_rank', 0)
    map_s = data.get('ranking_metrics',{}).get('map_score', 0)
    ndcg = data.get('ranking_metrics',{}).get('ndcg_at_k',{}).get('10', 0)
    faith = data.get('generation_metrics',{}).get('evidence_faithfulness', 0)
    print(f'{dataset:<12} {tech:<12} {mrr:>8.4f} {map_s:>8.4f} {ndcg:>10.4f} {faith:>8.4f}')
