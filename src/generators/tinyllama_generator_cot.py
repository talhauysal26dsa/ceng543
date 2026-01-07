"""
TinyLlama Chain of Thought (CoT) generator.
Uses step-by-step reasoning prompts for improved answer quality.
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


class TinyLlamaGeneratorCoT:
    """
    TinyLlama generator with Chain of Thought prompting.
    
    Uses step-by-step reasoning to improve answer quality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TinyLlama CoT generator."""
        self.config = config
        self.model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = config.get("max_length", 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading TinyLlama (CoT mode): {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("TinyLlama CoT loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading TinyLlama: {e}")
            raise
    
    def generate(self, query: str, documents: List[Dict[str, Any]], 
                 max_tokens: int = None, temperature: float = None) -> GenerationResult:
        """Generate answer using Chain of Thought prompting."""
        import time
        
        evidence_docs = documents[:3]
        prompt = self._create_prompt(query, evidence_docs)
        
        try:
            start = time.time()
            answer = self._generate_text(prompt, max_tokens or 150, temperature or 0.7)
            gen_time = time.time() - start
            logger.info(f"CoT Generated in {gen_time:.2f}s")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "Error generating answer."
        
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
            confidence=0.80,  # CoT typically has higher confidence
            metadata={
                "model": "tinyllama-cot",
                "device": self.device,
                "prompt_mode": "chain_of_thought"
            }
        )
    
    def _create_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create Chain of Thought prompt."""
        evidence = ""
        for i, doc in enumerate(documents, 1):
            text = doc.get('text', '')[:200]
            evidence += f"[{i}] {text}\n"
        
        return f"""Question: {query}

Evidence:
{evidence}

Let me think through this step by step:

Step 1: Identify key information in the evidence.
Step 2: Connect the relevant facts to the question.
Step 3: Formulate a concise, evidence-based answer.

Based on my reasoning:
Answer:"""
    
    def _generate_text(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate text with TinyLlama."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
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
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer after "Answer:"
        if "Answer:" in full_text:
            parts = full_text.split("Answer:")
            answer = parts[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        # Clean up any reasoning artifacts
        if "Step" in answer:
            lines = answer.split('\n')
            answer = ' '.join(l for l in lines if not l.strip().startswith('Step'))
        
        return answer.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "prompt_mode": "chain_of_thought"
        }
