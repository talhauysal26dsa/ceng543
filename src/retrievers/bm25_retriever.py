"""
BM25 sparse retriever implementation.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 retrieval."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class BM25Retriever:
    """
    BM25 sparse retriever for document retrieval.
    
    Uses the BM25 algorithm to find relevant documents based on
    term frequency and document frequency statistics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BM25 retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.k1 = config.get("k1", 1.2)
        self.b = config.get("b", 0.75)
        self.bm25 = None
        self.documents = []
        self.document_ids = []
        
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for retrieval with disk caching.
        
        Args:
            documents: List of documents to index
        """
        import os
        import pickle
        import hashlib
        
        # Create cache directory
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache key from document count
        cache_key = hashlib.md5(str(len(documents)).encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"bm25_index_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading BM25 index from cache...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.documents = cached_data['documents']
                self.document_ids = cached_data['document_ids']
                self.bm25 = cached_data['bm25']
                
                logger.info(f"BM25 index loaded from cache! ({len(self.document_ids)} docs) - FAST!")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, rebuilding...")
        
        # Build index (original code)
        logger.info(f"Building BM25 index for {len(documents)} documents (SLOW - first time)...")
        
        self.documents = documents
        self.document_ids = []
        for i, doc in enumerate(documents):
            # Try to get ID from various fields
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("title") or i
            self.document_ids.append(doc_id)
        
        # Extract text content for indexing
        texts = []
        for doc in documents:
            text = doc.get("text", "")
            if isinstance(text, str):
                texts.append(text.split())
            else:
                texts.append([])
        
        # Initialize BM25 index
        self.bm25 = BM25Okapi(texts, k1=self.k1, b=self.b)
        
        # Save to cache
        try:
            cache_data = {
                'documents': self.documents,
                'document_ids': self.document_ids,
                'bm25': self.bm25
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"BM25 index saved to cache - next run will be FAST!")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        
        logger.info(f"Indexed {len(documents)} documents with BM25")
    
    def retrieve(self, query: str, max_documents: int = 100) -> BM25Result:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            max_documents: Maximum number of documents to return
            
        Returns:
            BM25Result: Retrieved documents and scores
        """
        if self.bm25 is None:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        # Tokenize query
        query_tokens = query.split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top documents
        top_indices = np.argsort(scores)[::-1][:max_documents]
        
        # Filter out documents with zero scores
        top_indices = [idx for idx in top_indices if scores[idx] > 0]
        
        # Prepare results
        retrieved_docs = []
        retrieved_scores = []
        
        for idx in top_indices:
            retrieved_docs.append(self.documents[idx])
            retrieved_scores.append(float(scores[idx]))
        
        return BM25Result(
            documents=retrieved_docs,
            scores=retrieved_scores,
            metadata={
                "retrieval_method": "bm25",
                "k1": self.k1,
                "b": self.b,
                "num_documents": len(retrieved_docs),
                "max_documents": max_documents
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        return {
            "num_documents": len(self.documents),
            "k1": self.k1,
            "b": self.b,
            "indexed": self.bm25 is not None
        }
