#!/bin/bash

# Download datasets for Multi-Agent RAG system
# This script downloads the required datasets for evaluation

set -e

# Create data directories
mkdir -p data/raw/ms_marco
mkdir -p data/raw/trec_dl
mkdir -p data/raw/hotpotqa
mkdir -p data/raw/fever

echo "Downloading datasets for Multi-Agent RAG system..."
echo ""

# MS MARCO requires registration - show instructions
echo "⚠️  MS MARCO Dataset requires registration:"
echo "   1. Go to: https://microsoft.github.io/msmarco/"
echo "   2. Click 'Download' and sign in with Microsoft account"
echo "   3. Accept the agreement"
echo "   4. Download these files to data/raw/ms_marco/:"
echo "      - queries.tsv"
echo "      - collection.tsv" 
echo "      - qrels.tsv"
echo ""

# Download HotpotQA dataset (should work)
echo "📥 Downloading HotpotQA dataset..."
cd data/raw/hotpotqa

# HotpotQA train/dev/test
wget -O train.json "https://hotpotqa.s3.amazonaws.com/hotpot_train_v1.1.json"
wget -O dev.json "https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json"
wget -O test.json "https://hotpotqa.s3.amazonaws.com/hotpot_test_v1.1.json"

cd ../../..

# Download FEVER dataset (should work)
echo "📥 Downloading FEVER dataset..."
cd data/raw/fever

# FEVER train/dev/test
wget -O train.json "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl"
wget -O dev.json "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl"
wget -O test.json "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl"

# FEVER Wikipedia dump
wget -O wiki-pages.zip "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip"
unzip wiki-pages.zip

cd ../../..

echo ""
echo "✅ Downloadable datasets completed!"
echo ""
echo "📊 Dataset sizes:"
echo "   - HotpotQA: ~50 MB"
echo "   - FEVER: ~200 MB"
echo "   - MS MARCO: ~2-3 GB (manual download required)"
echo "   - TREC-DL: ~500 MB (manual download required)"
echo ""
echo "📁 Datasets are available in:"
echo "   - HotpotQA: data/raw/hotpotqa/"
echo "   - FEVER: data/raw/fever/"
echo "   - MS MARCO: data/raw/ms_marco/ (after manual download)"
echo "   - TREC-DL: data/raw/trec_dl/ (after manual download)"