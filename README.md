# MA-RAG: Multi-Agent Retrieval-Augmented Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Academic implementation of a modular multi-agent RAG system combining BM25, dense retrieval, and cross-encoder re-ranking.

## 🎯 Overview

MA-RAG demonstrates dataset-dependent performance gains:
- **MS MARCO** (noisy web): +207% MRR improvement
- **HotpotQA** (multi-hop): +17% MRR improvement  
- **FEVER** (fact verification): +3.5% MRR improvement (0.96 → 0.99)

---

## 🚀 Reproduction Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/talhauysal26dsa/ceng543.git
cd ceng543
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download and Prepare Datasets

Run the data download script:
```bash
python scripts/download_hf_datasets.py
```

This will:
- Download MS MARCO, HotpotQA, FEVER from HuggingFace 
- Create test sets (100 queries each)
- Preprocess documents
- Save to `data/test/` and `data/processed/`

**Manual Alternative**: Place datasets in:
- `data/test/ms_marco_test_100.json`
- `data/test/hotpotqa_test_100.json`
- `data/test/fever_test_100.json`
- `data/processed/ms_marco_documents.json`
- `data/processed/hotpotqa_documents.json`
- `data/processed/fever_test_docs.json`

### Step 5: Run Prompting Ablation Study

Reproduce all results (9 experiments: 3 datasets × 3 techniques):

```bash
python run_prompt_ablation.py
```

This will:
- Run Baseline, CoT, and Few-Shot prompting
- Test on MS MARCO, HotpotQA, and FEVER
- Save results to `experiments/results/prompt_ablation_*.json`

### Step 6: View Results

```bash
python show_prompt_results.py
```

**Expected Runtime**: ~2-3 hours on GPU, ~8-10 hours on CPU

---

## 📊 Expected Results

### Architectural Ablation

| Dataset | Config | MRR | MAP | nDCG@10 |
|---------|--------|-----|-----|---------|
| **MS MARCO** | BM25 Only | 0.182 | 0.182 | 0.189 |
| | +Dense | 0.360 | 0.363 | 0.478 |
| | +Cross-Encoder | **0.558** | **0.559** | **0.708** |
| **HotpotQA** | BM25 Only | 0.768 | 0.762 | 0.792 |
| | +Dense | 0.822 | 0.823 | 1.153 |
| | +Cross-Encoder | **0.900** | **0.898** | **1.309** |
| **FEVER** | BM25 Only | 0.960 | 0.960 | 0.963 |
| | +Dense | 0.975 | 0.977 | 1.565 |
| | +Cross-Encoder | **0.993** | **0.993** | **1.591** |

### Prompting Ablation (TinyLlama-1.1B)

| Dataset | Baseline | CoT | Few-Shot | Best Δ |
|---------|----------|-----|----------|--------|
| FEVER | 0.574 | 0.580 | **0.587** | +2.2% |
| HotpotQA | **0.710** | 0.630 | 0.700 | -1.4% |
| MS MARCO | **0.665** | 0.608 | 0.664 | -0.2% |

---

## 🏗️ Architecture

```
SUPERVISOR AGENT
(Pipeline Orchestration)
       │
       ├─► RETRIEVER (BM25 + Dense Embeddings)
       ├─► RANKER (Cross-Encoder Re-ranking)
       └─► SUMMARIZER (TinyLlama + Prompting)
```

## 📁 Project Structure

```
ceng543/
├── src/
│   ├── agents/          # Supervisor, Retriever, Ranker, Summarizer
│   ├── generators/      # TinyLlama (baseline, CoT, few-shot)
│   ├── evaluation/      # Metrics (MRR, MAP, nDCG, Faithfulness)
│   └── pipeline/        # MA-RAG pipeline
├── config/
│   ├── agent_config.yaml         # Baseline config
│   ├── agent_config_cot.yaml     # Chain of Thought
│   └── agent_config_fewshot.yaml # Few-Shot Learning
├── data/
│   ├── test/            # Test datasets (100 queries each)
│   └── processed/       # Preprocessed documents
├── experiments/
│   └── results/         # JSON results
├── scripts/
│   └── download_hf_datasets.py  # Data download script
├── run_prompt_ablation.py
├── show_prompt_results.py
└── requirements.txt
```

## 🔬 Technical Details

**Models Used:**
- Dense Retrieval: `all-mpnet-base-v2` (768-dim embeddings)
- Re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generation: `TinyLlama-1.1B-Chat-v1.0`

**Evaluation Metrics:**
- Retrieval: MRR, MAP, nDCG@k, Recall@k, Precision@k
- Generation: ROUGE-L, BLEU, Evidence Faithfulness

**Reproducibility:**
- Fixed random seeds (42)
- 100 queries per dataset
- Statistical significance: paired t-test (p < 0.05)

## 📝 Citation

```bibtex
@software{marag2026,
  author = {Talha Uysal},
  title = {MA-RAG: Multi-Agent Retrieval-Augmented Generation},
  year = {2026},
  url = {https://github.com/talhauysal26dsa/ceng543}
}
```

## 📄 License

MIT License
