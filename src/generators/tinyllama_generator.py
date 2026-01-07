"""
TinyLlama local LLM generator using Hugging Face Transformers.
Lightweight model suitable for 4GB RAM systems.
"""

from typing import Any, Dict, List
import logging
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class TinyLlamaGenerator:
    """
    TinyLlama 1.1B generator for local inference.
    
    Optimized for low RAM usage (2-3GB).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TinyLlama generator."""
        self.config = config
        self.model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = config.get("max_length", 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading TinyLlama: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer (offline mode - use cached)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            
            # Load model - low memory mode (offline - use cached)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("TinyLlama loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading TinyLlama: {e}")
            raise
    
    def generate(self, query: str, documents: List[Dict[str, Any]], 
                 max_tokens: int = None, temperature: float = None) -> GenerationResult:
        """Generate answer using TinyLlama."""
        import time
        
        # Top 3 docs
        evidence_docs = documents[:3]
        
        # Build prompt
        prompt = self._create_prompt(query, evidence_docs)
        
        # Generate
        try:
            start = time.time()
            answer = self._generate_text(prompt, max_tokens or 100, temperature or 0.7)
            gen_time = time.time() - start
            logger.info(f"Generated in {gen_time:.2f}s")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "Error generating answer."
        
        # Evidence
        evidence = []
        for i, doc in enumerate(evidence_docs, 1):
            evidence.append({
                'text': doc.get('text', '')[:200],
                'relevance_score': 1.0 - (i * 0.1),
                'source': doc.get('id', doc.get('doc_id', f'doc_{i}'))
            })
        
        return GenerationResult(
            answer=answer.strip(),
            evidence=evidence,
            confidence=0.75,
            metadata={
                "model": "tinyllama",
                "device": self.device
            }
        )
    
    def _create_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create prompt."""
        evidence = ""
        for i, doc in enumerate(documents, 1):
            text = doc.get('text', '')[:200]
            evidence += f"[{i}] {text}\n"
        
        return f"""Question: {query}

Evidence:
{evidence}

Answer the question using the evidence. Be concise (2-3 sentences).

Answer:"""
    
    def _generate_text(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate text with TinyLlama."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after "Answer:")
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        return answer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length
        }
