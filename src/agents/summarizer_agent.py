"""
Summarizer agent for Multi-Agent RAG system.
Generates evidence-grounded answers from ranked documents.
"""

from typing import Any, Dict, List, Optional
import logging
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult
from ..generators.summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from answer generation."""
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class SummarizerAgent(BaseAgent):
    """
    Agent responsible for generating evidence-grounded answers.
    
    Takes ranked documents and generates coherent, well-supported
    answers using the retrieved evidence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the summarizer agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("summarizer_agent", config)
        
        # Check mode priority: tinyllama > api > local
        use_tinyllama = config.get("use_tinyllama", False)
        use_api = config.get("use_api", False)
        prompt_mode = config.get("prompt_mode", "baseline")  # baseline, cot, fewshot
        
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        
        if use_tinyllama:
            # Select TinyLlama variant based on prompt_mode
            tinyllama_config = config.get("tinyllama_config", {})
            
            if prompt_mode == "cot":
                from ..generators.tinyllama_generator_cot import TinyLlamaGeneratorCoT
                self.summarizer = TinyLlamaGeneratorCoT(tinyllama_config)
                logger.info("SummarizerAgent using TinyLlama (Chain of Thought)")
            elif prompt_mode == "fewshot":
                from ..generators.tinyllama_generator_fewshot import TinyLlamaGeneratorFewShot
                self.summarizer = TinyLlamaGeneratorFewShot(tinyllama_config)
                logger.info("SummarizerAgent using TinyLlama (Few-Shot Learning)")
            else:
                from ..generators.tinyllama_generator import TinyLlamaGenerator
                self.summarizer = TinyLlamaGenerator(tinyllama_config)
                logger.info("SummarizerAgent using TinyLlama (baseline)")
        elif use_api:
            # Use Gemini API
            from ..generators.gemini_generator import GeminiGenerator
            api_config = config.get("api_config", {})
            self.summarizer = GeminiGenerator(api_config)
            logger.info("SummarizerAgent initialized with Gemini API")
        else:
            # Use local model
            from ..generators.summarizer import Summarizer
            model_config = config.get("model_config", {})
            self.summarizer = Summarizer(model_config)
            logger.info("SummarizerAgent initialized with local model")
        
    def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Generate answer from ranked documents.
        
        Args:
            input_data: Dictionary containing 'query' and 'documents'
            **kwargs: Additional parameters
            
        Returns:
            AgentResult: Generated answer and evidence
        """
        try:
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    data=None,
                    metadata={},
                    error="Invalid input data"
                )
            
            # Extract query and documents
            query = input_data["query"]
            documents = input_data["documents"]
            
            # Generate answer using summarizer
            generation_result = self.summarizer.generate(
                query=query,
                documents=documents,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result = AgentResult(
                success=True,
                data=generation_result,
                metadata={
                    "answer_length": len(generation_result.answer),
                    "num_evidence": len(generation_result.evidence),
                    "confidence": generation_result.confidence,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )
            
            self.log_result(result)
            return result
            
        except Exception as e:
            error_msg = f"Error in summarizer agent: {str(e)}"
            self.logger.error(error_msg)
            return AgentResult(
                success=False,
                data=None,
                metadata={},
                error=error_msg
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for generation.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid
        """
        if not isinstance(input_data, dict):
            return False
        
        if "query" not in input_data or "documents" not in input_data:
            return False
        
        query = input_data["query"]
        documents = input_data["documents"]
        
        if not isinstance(query, str) or len(query.strip()) == 0:
            return False
        
        if not isinstance(documents, list) or len(documents) == 0:
            return False
        
        # Check that documents have required fields
        for doc in documents:
            if not isinstance(doc, dict) or "text" not in doc:
                return False
        
        return True
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generation performance."""
        return {
            "model_name": self.summarizer.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
