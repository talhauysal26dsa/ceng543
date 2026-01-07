"""
Data loader implementation for Multi-Agent RAG system.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    num_queries: int
    num_documents: int
    file_paths: Dict[str, str]


class DataLoader:
    """
    Data loader for various RAG datasets.
    
    Supports loading and preprocessing of:
    - MS MARCO
    - TREC Deep Learning Track
    - HotpotQA
    - FEVER
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config.get("data_dir", "data"))
        
    def load_dataset(self, dataset_name: str) -> DatasetInfo:
        """
        Load a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            DatasetInfo: Information about the loaded dataset
        """
        if dataset_name == "ms_marco":
            return self._load_ms_marco()
        elif dataset_name == "trec_dl":
            return self._load_trec_dl()
        elif dataset_name == "hotpotqa":
            return self._load_hotpotqa()
        elif dataset_name == "fever":
            return self._load_fever()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_ms_marco(self) -> DatasetInfo:
        """Load MS MARCO dataset."""
        dataset_dir = self.data_dir / "raw" / "ms_marco"
        
        # Load queries
        queries_file = dataset_dir / "queries.tsv"
        if queries_file.exists():
            queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['id', 'text'])
            queries = queries_df.to_dict('records')
        else:
            queries = []
        
        # Load collection
        collection_file = dataset_dir / "collection.tsv"
        if collection_file.exists():
            collection_df = pd.read_csv(collection_file, sep='\t', header=None, names=['id', 'text'])
            documents = collection_df.to_dict('records')
        else:
            documents = []
        
        # Load qrels
        qrels_file = dataset_dir / "qrels.tsv"
        if qrels_file.exists():
            qrels_df = pd.read_csv(qrels_file, sep='\t', header=None, names=['query_id', 'doc_id', 'relevance'])
            qrels = qrels_df.to_dict('records')
        else:
            qrels = []
        
        return DatasetInfo(
            name="MS MARCO",
            description="Microsoft Machine Reading Comprehension dataset",
            num_queries=len(queries),
            num_documents=len(documents),
            file_paths={
                "queries": str(queries_file),
                "collection": str(collection_file),
                "qrels": str(qrels_file)
            }
        )
    
    def _load_trec_dl(self) -> DatasetInfo:
        """Load TREC Deep Learning Track dataset."""
        dataset_dir = self.data_dir / "raw" / "trec_dl"
        
        # Load train data
        train_file = dataset_dir / "train.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_data = json.load(f)
        else:
            train_data = []
        
        # Load dev data
        dev_file = dataset_dir / "dev.json"
        if dev_file.exists():
            with open(dev_file, 'r') as f:
                dev_data = json.load(f)
        else:
            dev_data = []
        
        # Load test data
        test_file = dataset_dir / "test.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = []
        
        # Load collection
        collection_file = dataset_dir / "collection.tsv"
        if collection_file.exists():
            collection_df = pd.read_csv(collection_file, sep='\t', header=None, names=['id', 'text'])
            documents = collection_df.to_dict('records')
        else:
            documents = []
        
        return DatasetInfo(
            name="TREC Deep Learning Track",
            description="TREC Deep Learning Track dataset",
            num_queries=len(train_data) + len(dev_data) + len(test_data),
            num_documents=len(documents),
            file_paths={
                "train": str(train_file),
                "dev": str(dev_file),
                "test": str(test_file),
                "collection": str(collection_file)
            }
        )
    
    def _load_hotpotqa(self) -> DatasetInfo:
        """Load HotpotQA dataset."""
        dataset_dir = self.data_dir / "raw" / "hotpotqa"
        
        # Load train data
        train_file = dataset_dir / "train.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_data = json.load(f)
        else:
            train_data = []
        
        # Load dev data
        dev_file = dataset_dir / "dev.json"
        if dev_file.exists():
            with open(dev_file, 'r') as f:
                dev_data = json.load(f)
        else:
            dev_data = []
        
        # Load test data
        test_file = dataset_dir / "test.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = []
        
        return DatasetInfo(
            name="HotpotQA",
            description="Multi-hop reasoning dataset",
            num_queries=len(train_data) + len(dev_data) + len(test_data),
            num_documents=0,  # HotpotQA doesn't have a separate document collection
            file_paths={
                "train": str(train_file),
                "dev": str(dev_file),
                "test": str(test_file)
            }
        )
    
    def _load_fever(self) -> DatasetInfo:
        """Load FEVER dataset."""
        dataset_dir = self.data_dir / "raw" / "fever"
        
        # Load train data
        train_file = dataset_dir / "train.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_data = json.load(f)
        else:
            train_data = []
        
        # Load dev data
        dev_file = dataset_dir / "dev.json"
        if dev_file.exists():
            with open(dev_file, 'r') as f:
                dev_data = json.load(f)
        else:
            dev_data = []
        
        # Load test data
        test_file = dataset_dir / "test.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = []
        
        return DatasetInfo(
            name="FEVER",
            description="Fact verification dataset",
            num_queries=len(train_data) + len(dev_data) + len(test_data),
            num_documents=0,  # FEVER doesn't have a separate document collection
            file_paths={
                "train": str(train_file),
                "dev": str(dev_file),
                "test": str(test_file)
            }
        )
    
    def preprocess_documents(self, documents: List[Dict[str, Any]], 
                           config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Preprocess documents according to configuration.
        
        Args:
            documents: List of documents to preprocess
            config: Preprocessing configuration
            
        Returns:
            List[Dict[str, Any]]: Preprocessed documents
        """
        processed_docs = []
        
        for doc in documents:
            processed_doc = doc.copy()
            
            # Clean text if configured
            if config.get("text_cleaning", True):
                processed_doc["text"] = self._clean_text(doc.get("text", ""))
            
            # Filter by length if configured
            min_length = config.get("min_document_length", 0)
            max_length = config.get("max_document_length", float('inf'))
            
            text_length = len(processed_doc.get("text", "").split())
            if min_length <= text_length <= max_length:
                processed_docs.append(processed_doc)
        
        # Remove duplicates if configured
        if config.get("remove_duplicates", True):
            processed_docs = self._remove_duplicates(processed_docs)
        
        logger.info(f"Preprocessed {len(documents)} documents to {len(processed_docs)} documents")
        
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text.strip()
    
    def _remove_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on text content."""
        seen_texts = set()
        unique_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            if text not in seen_texts:
                seen_texts.add(text)
                unique_docs.append(doc)
        
        return unique_docs
    
    def save_processed_data(self, data: Any, output_path: str) -> None:
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            output_path: Path to save data
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif output_path.endswith('.csv'):
            if isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False)
            else:
                raise ValueError("Data must be a list of dictionaries for CSV format")
        else:
            raise ValueError(f"Unsupported file format: {output_file.suffix}")
        
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: str) -> Any:
        """
        Load processed data from file.
        
        Args:
            input_path: Path to load data from
            
        Returns:
            Any: Loaded data
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        if input_path.endswith('.json'):
            with open(input_file, 'r') as f:
                return json.load(f)
        elif input_path.endswith('.csv'):
            return pd.read_csv(input_file).to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
