"""
Gemini API-based generator for answer generation.
"""

from typing import Any, Dict, List, Optional
import logging
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class GeminiGenerator:
    """
    Gemini API-based generator for evidence-grounded answers.
    
    Uses Gemini API instead of local LLM for answer generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini generator.
        
        Args:
            config: Configuration dictionary with api_key, model, etc.
        """
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gemini-2.5-flash")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        
        if not self.api_key:
            raise ValueError("Gemini API key is required in config")
        
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        logger.info(f"Initialized GeminiGenerator with model: {self.model}")
    
    def generate(self, query: str, documents: List[Dict[str, Any]], 
                 max_tokens: int = None, temperature: float = None) -> GenerationResult:
        """
        Generate answer from ranked documents using Gemini API.
        
        Args:
            query: Query string
            documents: List of ranked documents
            max_tokens: Override max tokens (optional)
            temperature: Override temperature (optional)
            
        Returns:
            GenerationResult: Generated answer and evidence
        """
        import time
        
        # Use provided params or defaults
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        # Prepare evidence from documents
        evidence_docs = documents[:5]  # Use top 5
        
        # Build prompt
        prompt = self._create_prompt(query, evidence_docs)
        
        # Call Gemini API
        try:
            answer = self._call_gemini_api(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            answer = "I apologize, but I encountered an error while generating the answer."
        
        # RATE LIMITING: Wait 6 seconds to stay under 15 req/min
        time.sleep(6)
        
        # Extract evidence
        evidence = self._extract_evidence(evidence_docs, answer)
        
        # Calculate confidence
        confidence = self._calculate_confidence(answer, evidence)
        
        return GenerationResult(
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            metadata={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "num_documents": len(documents)
            }
        )
    
    def _create_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Create prompt for Gemini API.
        
        Args:
            query: Query string
            documents: Evidence documents
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""You are an AI assistant. Answer the question using the evidence below.

IMPORTANT: You MUST include citations [1], [2], [3] etc. in your answer!

Question: {query}

Evidence:
"""
        
        for i, doc in enumerate(documents, 1):
            # Limit doc length
            doc_text = doc.get('text', '')[:400]
            prompt += f"\n[{i}] {doc_text}\n"
        
        prompt += """
RULES:
1. Answer in 2-3 sentences maximum
2. MUST include [1], [2], [3] citations for every fact
3. Format example: "The project began in 1942 [1] and involved 130,000 people [2]."
4. If no answer in evidence, say: "The evidence does not answer this question."

Answer with citations:"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Call Gemini API.
        
        Args:
            prompt: Input prompt
            max_tokens: Max output tokens
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        response = requests.post(
            self.api_url,
            headers={
                'Content-Type': 'application/json',
                'X-goog-api-key': self.api_key
            },
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise RuntimeError(f"API error {response.status_code}: {response.text[:200]}")
    
    def _extract_evidence(self, documents: List[Dict[str, Any]], answer: str) -> List[Dict[str, Any]]:
        """
        Extract evidence from documents.
        
        Args:
            documents: Source documents
            answer: Generated answer
            
        Returns:
            List[Dict]: Evidence with scores
        """
        evidence = []
        
        for i, doc in enumerate(documents[:3], 1):
            text = doc.get('text', '')
            
            # Simple relevance: check if citation is in answer
            citation_used = f'[{i}]' in answer
            relevance_score = 1.0 if citation_used else 0.5
            
            evidence.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'relevance_score': relevance_score,
                'source': doc.get('id', doc.get('doc_id', 'unknown'))
            })
        
        return evidence
    
    def _calculate_confidence(self, answer: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score.
        
        Args:
            answer: Generated answer
            evidence: Evidence documents
            
        Returns:
            float: Confidence score (0-1)
        """
        if not evidence:
            return 0.0
        
        # Check for citations
        citations_found = sum(1 for i in range(1, 10) if f'[{i}]' in answer)
        citation_score = min(citations_found / 3.0, 1.0)
        
        # Average evidence relevance
        avg_relevance = sum(e['relevance_score'] for e in evidence) / len(evidence)
        
        # Combine
        confidence = (citation_score * 0.6) + (avg_relevance * 0.4)
        
        return min(confidence, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the generator."""
        return {
            "model": self.model,
            "api_url": self.api_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
