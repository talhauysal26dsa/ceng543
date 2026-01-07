"""
Summarizer implementation for evidence-grounded answer generation.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class Summarizer:
    """
    Summarizer for generating evidence-grounded answers.
    
    Takes ranked documents and generates coherent, well-supported
    answers using the retrieved evidence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize summarizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get("model_name", "microsoft/DialoGPT-medium")
        self.max_length = config.get("max_length", 1024)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def generate(self, query: str, documents: List[Dict[str, Any]], 
                 max_tokens: int = 512, temperature: float = 0.7) -> GenerationResult:
        """
        Generate answer from ranked documents.
        
        Args:
            query: Query string
            documents: List of ranked documents
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            GenerationResult: Generated answer and evidence
        """
        # Prepare context from documents
        context = self._prepare_context(documents)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate answer
        generated_text = self._generate_text(prompt, max_tokens, temperature)
        
        # Extract answer and evidence
        answer = self._extract_answer(generated_text, query)
        evidence = self._extract_evidence(documents, answer)
        confidence = self._calculate_confidence(answer, evidence)
        
        return GenerationResult(
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "num_documents": len(documents)
            }
        )
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare context from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            str: Formatted context
        """
        context_parts = []
        
        for i, doc in enumerate(documents[:5]):  # Use top 5 documents
            text = doc.get("text", "")
            if text:
                context_parts.append(f"Document {i+1}: {text[:500]}...")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for generation.
        
        Args:
            query: Query string
            context: Context from documents
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        try:
            # Truncate prompt if too long
            max_prompt_length = self.max_length - max_tokens
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length]
            
            # Generate text
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            return result[0]["generated_text"]
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def _extract_answer(self, generated_text: str, query: str) -> str:
        """
        Extract answer from generated text.
        
        Args:
            generated_text: Generated text
            query: Original query
            
        Returns:
            str: Extracted answer
        """
        # Find the answer part after "Answer:"
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            # Fallback: use the text after the prompt
            answer = generated_text[len(query):].strip()
        
        # Clean up the answer
        answer = answer.replace("\n", " ").strip()
        
        return answer if answer else "I couldn't generate a proper answer based on the provided context."
    
    def _extract_evidence(self, documents: List[Dict[str, Any]], answer: str) -> List[Dict[str, Any]]:
        """
        Extract relevant evidence from documents.
        
        Args:
            documents: List of documents
            answer: Generated answer
            
        Returns:
            List[Dict[str, Any]]: Evidence with relevance scores
        """
        evidence = []
        
        for doc in documents[:3]:  # Use top 3 documents as evidence
            text = doc.get("text", "")
            if text:
                # Simple relevance scoring based on word overlap
                relevance_score = self._calculate_relevance(answer, text)
                
                evidence.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "relevance_score": relevance_score,
                    "source": doc.get("id", "unknown")
                })
        
        return evidence
    
    def _calculate_relevance(self, answer: str, text: str) -> float:
        """
        Calculate relevance score between answer and text.
        
        Args:
            answer: Generated answer
            text: Document text
            
        Returns:
            float: Relevance score (0-1)
        """
        answer_words = set(answer.lower().split())
        text_words = set(text.lower().split())
        
        if not answer_words or not text_words:
            return 0.0
        
        overlap = len(answer_words.intersection(text_words))
        union = len(answer_words.union(text_words))
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_confidence(self, answer: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for the answer.
        
        Args:
            answer: Generated answer
            evidence: Evidence from documents
            
        Returns:
            float: Confidence score (0-1)
        """
        if not evidence:
            return 0.0
        
        # Calculate average relevance of evidence
        avg_relevance = sum(e["relevance_score"] for e in evidence) / len(evidence)
        
        # Factor in answer length (longer answers might be more confident)
        length_factor = min(len(answer.split()) / 50, 1.0)
        
        # Combine factors
        confidence = (avg_relevance * 0.7) + (length_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the summarizer."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "device": str(self.device)
        }
