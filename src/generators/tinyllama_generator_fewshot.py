"""
TinyLlama Few-Shot Learning generator.
Uses 3 example input-output pairs for in-context learning.
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


# Few-shot examples for in-context learning
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "evidence": "[1] Paris is the capital and most populous city of France, with a population of over 2 million.",
        "answer": "Paris is the capital of France [1]."
    },
    {
        "question": "When was the first iPhone released?",
        "evidence": "[1] Apple Inc. introduced the first iPhone on January 9, 2007, at the Macworld Conference.",
        "answer": "The first iPhone was released on January 9, 2007 [1]."
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "evidence": "[1] Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
        "answer": "William Shakespeare wrote Romeo and Juliet [1]."
    }
]


class TinyLlamaGeneratorFewShot:
    """
    TinyLlama generator with Few-Shot Learning (3 examples).
    
    Uses in-context learning with example question-answer pairs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TinyLlama Few-Shot generator."""
        self.config = config
        self.model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.max_length = config.get("max_length", 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_shots = config.get("num_shots", 3)
        
        logger.info(f"Loading TinyLlama (Few-Shot mode, {self.num_shots} examples): {self.model_name}")
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
            
            logger.info("TinyLlama Few-Shot loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading TinyLlama: {e}")
            raise
    
    def generate(self, query: str, documents: List[Dict[str, Any]], 
                 max_tokens: int = None, temperature: float = None) -> GenerationResult:
        """Generate answer using Few-Shot Learning."""
        import time
        
        evidence_docs = documents[:3]
        prompt = self._create_prompt(query, evidence_docs)
        
        try:
            start = time.time()
            answer = self._generate_text(prompt, max_tokens or 100, temperature or 0.7)
            gen_time = time.time() - start
            logger.info(f"Few-Shot Generated in {gen_time:.2f}s")
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
            confidence=0.82,  # Few-shot typically has good confidence
            metadata={
                "model": "tinyllama-fewshot",
                "device": self.device,
                "prompt_mode": "few_shot",
                "num_shots": self.num_shots
            }
        )
    
    def _create_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create Few-Shot prompt with examples."""
        # Build examples section
        examples_text = "Here are examples of how to answer questions using evidence:\n\n"
        
        for i, example in enumerate(FEW_SHOT_EXAMPLES[:self.num_shots], 1):
            examples_text += f"""Example {i}:
Question: {example['question']}
Evidence: {example['evidence']}
Answer: {example['answer']}

"""
        
        # Build current query evidence
        evidence = ""
        for i, doc in enumerate(documents, 1):
            text = doc.get('text', '')[:200]
            evidence += f"[{i}] {text}\n"
        
        return f"""{examples_text}Now answer this question using the same format:

Question: {query}

Evidence:
{evidence}

Answer:"""
    
    def _generate_text(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate text with TinyLlama."""
        # Few-shot prompts are longer, so truncate more aggressively
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
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
        
        # Extract answer after final "Answer:"
        if "Answer:" in full_text:
            parts = full_text.split("Answer:")
            answer = parts[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        # Clean up - remove any new examples that might have been generated
        if "Example" in answer or "Question:" in answer:
            answer = answer.split("Example")[0].split("Question:")[0].strip()
        
        return answer.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "prompt_mode": "few_shot",
            "num_shots": self.num_shots
        }
