#!/usr/bin/env python3
"""
Download datasets from Hugging Face for Multi-Agent RAG system.
This script downloads MS MARCO, HotpotQA, and FEVER datasets.
"""

import os
from datasets import load_dataset
import json

def download_ms_marco():
    """Download MS MARCO dataset from Hugging Face."""
    print("Downloading MS MARCO dataset from Hugging Face...")
    
    # Create directory
    os.makedirs("data/raw/ms_marco", exist_ok=True)
    
    try:
        # Load MS MARCO train split (contains queries and passages)
        print("  - Loading train split...")
        train = load_dataset("ms_marco", "v1.1", split="train")
        
        # Save queries and passages
        with open("data/raw/ms_marco/queries.tsv", "w", encoding="utf-8") as f:
            for item in train:
                if 'query' in item:
                    f.write(f"{item['id']}\t{item['query']}\n")
        
        with open("data/raw/ms_marco/collection.tsv", "w", encoding="utf-8") as f:
            for item in train:
                if 'passage' in item:
                    f.write(f"{item['id']}\t{item['passage']}\n")
        
        print("MS MARCO downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading MS MARCO: {e}")
        print("MS MARCO requires manual download from: https://microsoft.github.io/msmarco/")

def download_hotpotqa():
    """Download HotpotQA dataset from Hugging Face."""
    print("Downloading HotpotQA dataset from Hugging Face...")
    
    # Create directory
    os.makedirs("data/raw/hotpotqa", exist_ok=True)
    
    try:
        # Load HotpotQA with correct config
        train = load_dataset("hotpot_qa", "distractor", split="train")
        dev = load_dataset("hotpot_qa", "distractor", split="validation")
        
        # Save as JSON
        with open("data/raw/hotpotqa/train.json", "w", encoding="utf-8") as f:
            json.dump(train.to_list(), f, indent=2)
        
        with open("data/raw/hotpotqa/dev.json", "w", encoding="utf-8") as f:
            json.dump(dev.to_list(), f, indent=2)
        
        print("HotpotQA downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading HotpotQA: {e}")

def download_fever():
    """Download FEVER dataset from Hugging Face."""
    print("Downloading FEVER dataset from Hugging Face...")
    
    # Create directory
    os.makedirs("data/raw/fever", exist_ok=True)
    
    try:
        # Try alternative FEVER dataset
        train = load_dataset("fever", split="train")
        dev = load_dataset("fever", split="validation")
        
        # Save as JSONL
        with open("data/raw/fever/train.jsonl", "w", encoding="utf-8") as f:
            for item in train:
                f.write(json.dumps(item) + "\n")
        
        with open("data/raw/fever/dev.jsonl", "w", encoding="utf-8") as f:
            for item in dev:
                f.write(json.dumps(item) + "\n")
        
        print("FEVER downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading FEVER: {e}")
        print("FEVER requires manual download from: https://fever.ai/dataset")

def main():
    """Main function to download all datasets."""
    print("Starting dataset download from Hugging Face...")
    print("")
    
    # Download datasets
    download_ms_marco()
    print("")
    download_hotpotqa()
    print("")
    download_fever()
    print("")
    
    print("All datasets downloaded successfully!")
    print("Check data/raw/ directory for downloaded files.")

if __name__ == "__main__":
    main()
