"""
Dense retriever implementation using sentence transformers.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DenseResult:
    """Result from dense retrieval."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]


class DenseRetriever:
    """
    Dense retriever using sentence transformers for semantic similarity.
    
    Uses pre-trained sentence transformers to encode queries and documents
    into dense vectors and retrieves based on cosine similarity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dense retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        
        # Initialize model
        self.model = SentenceTransformer(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Document embeddings cache
        self.document_embeddings = None
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
        
        # Create cache key from document count and model name
        cache_key = hashlib.md5(f"{len(documents)}_{self.model_name}".encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"dense_embeddings_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading Dense embeddings from cache...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.documents = cached_data['documents']
                self.document_ids = cached_data['document_ids']
                self.document_embeddings = cached_data['embeddings']
                
                logger.info(f"Dense embeddings loaded from cache! ({len(self.document_ids)} docs) - FAST!")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, rebuilding...")
        
        # Build embeddings (original code)
        logger.info(f"Encoding {len(documents)} documents with Dense retriever (SLOW - first time)...")
        
        self.documents = documents
        self.document_ids = []
        for i, doc in enumerate(documents):
            # Try to get ID from various fields
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("title") or i
            self.document_ids.append(doc_id)
        
        # Extract text content
        texts = [doc.get("text", "") for doc in documents]
        
        # Encode documents in batches
        self.document_embeddings = self._encode_texts(texts)
        
        # Save to cache
        try:
            cache_data = {
                'documents': self.documents,
                'document_ids': self.document_ids,
                'embeddings': self.document_embeddings
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Dense embeddings saved to cache - next run will be FAST!")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        
        logger.info(f"Indexed {len(documents)} documents with dense retriever")
    
    def retrieve(self, query: str, max_documents: int = 100) -> DenseResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            max_documents: Maximum number of documents to return
            
        Returns:
            DenseResult: Retrieved documents and scores
        """
        if self.document_embeddings is None:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        # Encode query
        query_embedding = self._encode_texts([query])
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding, 
            self.document_embeddings
        )[0]
        
        # Get top documents
        top_indices = np.argsort(similarities)[::-1][:max_documents]
        
        # Filter out documents with very low similarity
        top_indices = [idx for idx in top_indices if similarities[idx] > 0.1]
        
        # Prepare results
        retrieved_docs = []
        retrieved_scores = []
        
        for idx in top_indices:
            retrieved_docs.append(self.documents[idx])
            retrieved_scores.append(float(similarities[idx]))
        
        return DenseResult(
            documents=retrieved_docs,
            scores=retrieved_scores,
            metadata={
                "retrieval_method": "dense",
                "model_name": self.model_name,
                "num_documents": len(retrieved_docs),
                "max_documents": max_documents
            }
        )
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using the sentence transformer model.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            np.ndarray: Encoded embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        return {
            "num_documents": len(self.documents),
            "model_name": self.model_name,
            "embedding_dim": self.document_embeddings.shape[1] if self.document_embeddings is not None else None,
            "indexed": self.document_embeddings is not None
        }
