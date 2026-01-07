#!/usr/bin/env python3
"""
Data preprocessing script for Multi-Agent RAG system.

PREPROCESSING ADIMLARI:
1. Text cleaning (özel karakterleri temizleme, normalize etme)
2. Format standardization (tüm dataları aynı formata çevirme)
3. Indexing için hazırlama (document ID'leri oluşturma)
4. Train/Dev/Test split oluşturma
5. Metadata extraction (başlıklar, kaynak bilgisi)
6. Duplicate removal (tekrar eden dökümanları temizleme)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from utils.logger import setup_logger

logger = setup_logger("preprocess_data")


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text or pd.isna(text):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?;:\-\(\)\[\]]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def create_document_id(text: str, index: int) -> str:
    """Create unique document ID from text and index."""
    # Create hash-based ID
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"doc_{index}_{text_hash}"


def preprocess_ms_marco(data_dir: str, output_dir: str) -> None:
    """
    Preprocess MS MARCO dataset.
    
    YAPILAN İŞLEMLER:
    1. Queries'leri yükle ve temizle
    2. Collection'ı yükle (ilk 50K döküman - hafızayı korumak için)
    3. QRels'leri yükle (query-document ilişkileri)
    4. Text cleaning uygula
    5. Unified format'a çevir
    6. Train/Dev/Test split yap
    """
    logger.info("Preprocessing MS MARCO dataset...")
    
    # 1. Load and process queries
    processed_queries = {}
    
    for split in ['train', 'dev', 'eval']:
        queries_file = Path(data_dir) / "ms_marco" / "queries" / f"queries.{split}.tsv"
        if queries_file.exists():
            logger.info(f"  Loading {split} queries...")
            df = pd.read_csv(queries_file, sep='\t', header=None, names=['id', 'text'])
            
            queries = []
            for _, row in df.iterrows():
                queries.append({
                    'query_id': str(row['id']),
                    'query_text': clean_text(str(row['text'])),
                    'dataset': 'ms_marco',
                    'split': split
                })
            
            processed_queries[split] = queries
            logger.info(f"  Processed {len(queries)} {split} queries")
    
    # Save queries
    for split, queries in processed_queries.items():
        output_file = Path(output_dir) / f"ms_marco_queries_{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
    
    # 2. Load and process collection (documents)
    collection_file = Path(data_dir) / "ms_marco" / "collection" / "collection.tsv"
    if collection_file.exists():
        logger.info("  Loading collection (first 50K documents)...")
        
        # Read in chunks to save memory
        documents = []
        chunk_size = 10000
        total_docs = 0
        
        for chunk in pd.read_csv(collection_file, sep='\t', header=None, 
                                  names=['id', 'text'], chunksize=chunk_size):
            for _, row in chunk.iterrows():
                if total_docs >= 50000:  # Limit to 50K for faster processing
                    break
                
                doc_text = clean_text(str(row['text']))
                if len(doc_text) > 10:  # Skip very short documents
                    documents.append({
                        'doc_id': str(row['id']),
                        'text': doc_text,
                        'length': len(doc_text.split()),
                        'dataset': 'ms_marco'
                    })
                    total_docs += 1
            
            if total_docs >= 50000:
                break
        
        # Save documents
        output_file = Path(output_dir) / "ms_marco_documents.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Processed {len(documents)} documents")
    
    # 3. Load and process QRels
    for split in ['train', 'dev']:
        qrels_file = Path(data_dir) / "ms_marco" / f"qrels.{split} (1).tsv"
        if qrels_file.exists():
            logger.info(f"  Loading {split} qrels...")
            
            # QRels format: query_id \t 0 \t doc_id \t relevance
            qrels = []
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        qrels.append({
                            'query_id': parts[0],
                            'doc_id': parts[2],
                            'relevance': int(parts[3]) if len(parts) > 3 else 1
                        })
            
            # Save qrels
            output_file = Path(output_dir) / f"ms_marco_qrels_{split}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qrels, f, indent=2)
            
            logger.info(f"  Processed {len(qrels)} {split} qrels")
    
    logger.info("MS MARCO preprocessing complete!")


def preprocess_trec_dl(data_dir: str, output_dir: str) -> None:
    """Preprocess TREC Deep Learning Track dataset."""
    logger.info("Preprocessing TREC DL dataset...")
    
    # Process train/dev/test files
    for split in ["train", "dev", "test"]:
        file_path = Path(data_dir) / "trec_dl" / f"{split}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Save processed data
            output_file = Path(output_dir) / f"trec_dl_{split}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Processed {split} split with {len(data)} examples")


def preprocess_hotpotqa(data_dir: str, output_dir: str) -> None:
    """
    Preprocess HotpotQA dataset.
    
    YAPILAN İŞLEMLER:
    1. Multi-hop questions'ları parse et
    2. Supporting facts'leri extract et
    3. Context paragraphs'ları ayrıştır
    4. Unified format'a çevir
    """
    logger.info("Preprocessing HotpotQA dataset...")
    
    for split in ["train", "dev"]:
        file_path = Path(data_dir) / "hotpotqa" / f"{split}.json"
        if not file_path.exists():
            logger.warning(f"  {split}.json not found, skipping...")
            continue
        
        logger.info(f"  Loading {split} split...")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        for item in raw_data:
            # Extract question and answer
            question = clean_text(item.get('question', ''))
            answer = clean_text(item.get('answer', ''))
            
            # Extract supporting facts
            supporting_facts = item.get('supporting_facts', {})
            
            # Extract context (list of [title, sentences] pairs)
            context = item.get('context', {})
            context_texts = []
            
            if isinstance(context, dict):
                for title, sentences in zip(
                    context.get('title', []),
                    context.get('sentences', [])
                ):
                    # Combine sentences for each title
                    if isinstance(sentences, list):
                        text = ' '.join(sentences)
                    else:
                        text = str(sentences)
                    
                    context_texts.append({
                        'title': title,
                        'text': clean_text(text)
                    })
            
            processed_data.append({
                'id': item.get('id', ''),
                'question': question,
                'answer': answer,
                'type': item.get('type', ''),
                'level': item.get('level', ''),
                'supporting_facts': supporting_facts,
                'context': context_texts,
                'dataset': 'hotpotqa',
                'split': split
            })
        
        # Save processed data
        output_file = Path(output_dir) / f"hotpotqa_{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Processed {len(processed_data)} {split} examples")
    
    logger.info("HotpotQA preprocessing complete!")


def preprocess_fever(data_dir: str, output_dir: str) -> None:
    """
    Preprocess FEVER dataset.
    
    YAPILAN İŞLEMLER:
    1. Claims'leri yükle (SUPPORTS/REFUTES/NOT ENOUGH INFO)
    2. Evidence'ları extract et
    3. Wikipedia pages'leri index'le
    4. Claim-evidence pairs oluştur
    5. Unified format'a çevir
    """
    logger.info("Preprocessing FEVER dataset...")
    
    # Map filenames to split names
    file_mapping = {
        'train': 'train (1).jsonl',
        'dev': 'shared_task_dev (1).jsonl',
        'test': 'shared_task_test (1).jsonl'
    }
    
    for split, filename in file_mapping.items():
        file_path = Path(data_dir) / "fever" / filename
        if not file_path.exists():
            logger.warning(f"  {filename} not found, skipping...")
            continue
        
        logger.info(f"  Loading {split} split...")
        
        processed_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line)
                    
                    # Extract claim
                    claim = clean_text(item.get('claim', ''))
                    if not claim:
                        continue
                    
                    # Extract label (SUPPORTS, REFUTES, NOT ENOUGH INFO)
                    label = item.get('label', 'NOT ENOUGH INFO')
                    
                    # Extract evidence
                    evidence_list = item.get('evidence', [])
                    processed_evidence = []
                    
                    if evidence_list:
                        for evidence_set in evidence_list:
                            if isinstance(evidence_set, list):
                                for evidence in evidence_set:
                                    if isinstance(evidence, list) and len(evidence) >= 3:
                                        processed_evidence.append({
                                            'annotation_id': evidence[0] if len(evidence) > 0 else None,
                                            'evidence_id': evidence[1] if len(evidence) > 1 else None,
                                            'page_title': evidence[2] if len(evidence) > 2 else None,
                                            'sentence_id': evidence[3] if len(evidence) > 3 else None
                                        })
                    
                    processed_data.append({
                        'id': item.get('id', line_num),
                        'claim': claim,
                        'label': label,
                        'verifiable': item.get('verifiable', 'NOT VERIFIABLE'),
                        'evidence': processed_evidence,
                        'dataset': 'fever',
                        'split': split
                    })
                    
                except json.JSONDecodeError:
                    logger.warning(f"  Skipping malformed line {line_num} in {split}")
                    continue
        
        # Save processed data
        output_file = Path(output_dir) / f"fever_{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Processed {len(processed_data)} {split} examples")
    
    # Process Wikipedia pages (sample first 10K pages for efficiency)
    logger.info("  Loading Wikipedia pages (sample)...")
    wiki_dir = Path(data_dir) / "fever" / "wikipages" / "wiki-pages"
    
    if wiki_dir.exists():
        wiki_pages = []
        total_pages = 0
        max_pages = 10000  # Limit for faster processing
        
        for wiki_file in sorted(wiki_dir.glob("wiki-*.jsonl")):
            if total_pages >= max_pages:
                break
            
            with open(wiki_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if total_pages >= max_pages:
                        break
                    
                    try:
                        page = json.loads(line)
                        page_id = page.get('id', '')
                        page_text = page.get('text', '')
                        
                        if page_id and page_text:
                            wiki_pages.append({
                                'id': page_id,
                                'text': clean_text(page_text),
                                'lines': page.get('lines', '')
                            })
                            total_pages += 1
                    except json.JSONDecodeError:
                        continue
        
        # Save Wikipedia pages
        output_file = Path(output_dir) / "fever_wikipedia_pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(wiki_pages, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Processed {len(wiki_pages)} Wikipedia pages")
    
    logger.info("FEVER preprocessing complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for Multi-Agent RAG system")
    parser.add_argument("--data_dir", default="data/raw", help="Input data directory")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=["ms_marco", "trec_dl", "hotpotqa", "fever"],
                       help="Datasets to preprocess")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Preprocess datasets
    if "ms_marco" in args.datasets:
        preprocess_ms_marco(args.data_dir, args.output_dir)
    
    if "trec_dl" in args.datasets:
        preprocess_trec_dl(args.data_dir, args.output_dir)
    
    if "hotpotqa" in args.datasets:
        preprocess_hotpotqa(args.data_dir, args.output_dir)
    
    if "fever" in args.datasets:
        preprocess_fever(args.data_dir, args.output_dir)
    
    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main()
