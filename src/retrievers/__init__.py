"""
Retriever modules for Multi-Agent RAG system.
"""

from .bm25_retriever import BM25Retriever

# Try to import dense retriever (may fail due to dependency issues)
try:
    from .dense_retriever import DenseRetriever
except ImportError as e:
    print(f"Warning: Could not import DenseRetriever: {e}")
    DenseRetriever = None

# Try to import hybrid retriever (may fail due to dependency issues)
try:
    from .hybrid_retriever import HybridRetriever
except ImportError as e:
    print(f"Warning: Could not import HybridRetriever: {e}")
    HybridRetriever = None

__all__ = [
    "BM25Retriever",
    "DenseRetriever", 
    "HybridRetriever"
]
