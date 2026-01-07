"""
Generation metrics implementation for Multi-Agent RAG system.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetricsResult:
    """Result from generation metrics calculation."""
    rouge_l: float
    bleu_score: float
    evidence_faithfulness: float
    answer_length: int
    citation_quality: float
    metadata: Dict[str, Any]


class GenerationMetrics:
    """
    Metrics for evaluating generation performance.
    
    Implements standard generation metrics including:
    - ROUGE-L
    - BLEU score
    - Evidence faithfulness
    """
    
    def __init__(self):
        """Initialize generation metrics."""
        pass
        
    def calculate_metrics(self, generated_answers: List[str], 
                         reference_answers: List[str],
                         evidence_docs: List[List[Dict[str, Any]]]) -> GenerationMetricsResult:
        """
        Calculate generation metrics.
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference answers
            evidence_docs: List of evidence documents for each answer
            
        Returns:
            GenerationMetricsResult: Calculated metrics
        """
        if len(generated_answers) != len(reference_answers):
            raise ValueError("Number of generated answers must match reference answers")
        
        num_answers = len(generated_answers)
        
        # Calculate ROUGE-L scores
        rouge_l_scores = []
        for gen, ref in zip(generated_answers, reference_answers):
            rouge_l = self._calculate_rouge_l(gen, ref)
            rouge_l_scores.append(rouge_l)
        
        # Calculate BLEU scores
        bleu_scores = []
        for gen, ref in zip(generated_answers, reference_answers):
            bleu = self._calculate_bleu(gen, ref)
            bleu_scores.append(bleu)
        
        # Calculate evidence faithfulness
        faithfulness_scores = []
        for gen, evidence in zip(generated_answers, evidence_docs):
            faithfulness = self._calculate_evidence_faithfulness(gen, evidence)
            faithfulness_scores.append(faithfulness)
        
        # Calculate average answer length
        avg_length = np.mean([len(ans.split()) for ans in generated_answers])
        
        # Calculate citation quality
        citation_scores = []
        for gen, evidence in zip(generated_answers, evidence_docs):
            citation_quality = self._calculate_citation_quality(gen, evidence)
            citation_scores.append(citation_quality)
        
        return GenerationMetricsResult(
            rouge_l=np.mean(rouge_l_scores),
            bleu_score=np.mean(bleu_scores),
            evidence_faithfulness=np.mean(faithfulness_scores),
            answer_length=int(avg_length),
            citation_quality=np.mean(citation_scores),
            metadata={
                "num_answers": num_answers,
                "rouge_l_scores": rouge_l_scores,
                "bleu_scores": bleu_scores,
                "faithfulness_scores": faithfulness_scores,
                "citation_scores": citation_scores
            }
        )
    
    def _calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        Calculate ROUGE-L score.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            float: ROUGE-L score
        """
        generated_words = generated.split()
        reference_words = reference.split()
        
        if not generated_words or not reference_words:
            return 0.0
        
        # Calculate LCS
        lcs_length = self._lcs_length(generated_words, reference_words)
        
        # Calculate precision and recall
        precision = lcs_length / len(generated_words)
        recall = lcs_length / len(reference_words)
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calculate length of Longest Common Subsequence.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            int: LCS length
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_bleu(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            generated: Generated text
            reference: Reference text
            
        Returns:
            float: BLEU score
        """
        generated_words = generated.split()
        reference_words = reference.split()
        
        if not generated_words:
            return 0.0
        
        # Calculate n-gram precision for n=1 to 4
        precisions = []
        for n in range(1, 5):
            precision = self._calculate_ngram_precision(generated_words, reference_words, n)
            precisions.append(precision)
        
        # Calculate brevity penalty
        bp = min(1.0, len(generated_words) / len(reference_words)) if reference_words else 0.0
        
        # Calculate BLEU score
        if all(p > 0 for p in precisions):
            bleu = bp * (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
        else:
            bleu = 0.0
        
        return bleu
    
    def _calculate_ngram_precision(self, generated: List[str], reference: List[str], n: int) -> float:
        """
        Calculate n-gram precision.
        
        Args:
            generated: Generated words
            reference: Reference words
            n: N-gram size
            
        Returns:
            float: N-gram precision
        """
        if len(generated) < n:
            return 0.0
        
        # Generate n-grams
        gen_ngrams = [tuple(generated[i:i+n]) for i in range(len(generated)-n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
        
        if not gen_ngrams:
            return 0.0
        
        # Count matches
        gen_counts = Counter(gen_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        matches = 0
        for ngram, count in gen_counts.items():
            matches += min(count, ref_counts.get(ngram, 0))
        
        return matches / len(gen_ngrams)
    
    def _calculate_evidence_faithfulness(self, generated: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Calculate evidence faithfulness score.
        
        Args:
            generated: Generated answer
            evidence: Evidence documents
            
        Returns:
            float: Evidence faithfulness score
        """
        if not evidence:
            return 0.0
        
        # Extract text from evidence
        evidence_texts = [doc.get("text", "") for doc in evidence]
        all_evidence_text = " ".join(evidence_texts)
        
        # Calculate word overlap between generated answer and evidence
        generated_words = set(generated.lower().split())
        evidence_words = set(all_evidence_text.lower().split())
        
        if not generated_words:
            return 0.0
        
        # Calculate faithfulness as ratio of words from evidence
        words_from_evidence = generated_words.intersection(evidence_words)
        faithfulness = len(words_from_evidence) / len(generated_words)
        
        return faithfulness
    
    def _calculate_citation_quality(self, generated: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Calculate citation quality score based on explicit source tracking.
        
        Args:
            generated: Generated answer
            evidence: Evidence documents
            
        Returns:
            float: Citation quality score (0-1)
        """
        if not evidence:
            return 0.0
        
        # Look for citation markers like [1], (Source 1), etc.
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(Source \d+\)',  # (Source 1)
            r'\(Doc \d+\)',  # (Doc 1)
            r'\bdoc\d+\b',  # doc1, doc2, etc.
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            matches = re.findall(pattern, generated, re.IGNORECASE)
            total_citations += len(matches)
        
        # Citation quality is based on:
        # 1. Presence of citations (binary)
        # 2. Number of citations relative to evidence count
        has_citations = 1.0 if total_citations > 0 else 0.0
        citation_coverage = min(total_citations / len(evidence), 1.0) if evidence else 0.0
        
        # Weighted combination
        citation_quality = (has_citations * 0.5) + (citation_coverage * 0.5)
        
        return citation_quality
    
    def calculate_single_answer_metrics(self, generated: str, reference: str, 
                                      evidence: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate metrics for a single answer.
        
        Args:
            generated: Generated answer
            reference: Reference answer
            evidence: Evidence documents
            
        Returns:
            Dict[str, float]: Metrics for the answer
        """
        rouge_l = self._calculate_rouge_l(generated, reference)
        bleu = self._calculate_bleu(generated, reference)
        faithfulness = self._calculate_evidence_faithfulness(generated, evidence)
        citation_quality = self._calculate_citation_quality(generated, evidence)
        
        return {
            "rouge_l": rouge_l,
            "bleu": bleu,
            "faithfulness": faithfulness,
            "citation_quality": citation_quality,
            "length": len(generated.split())
        }
