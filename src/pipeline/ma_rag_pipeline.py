"""
Multi-Agent RAG Pipeline implementation.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import yaml
from pathlib import Path

from ..agents.supervisor import Supervisor
from ..utils.logger import setup_logger, LoggerMixin
from ..utils.data_loader import DataLoader

logger = setup_logger("ma_rag_pipeline")


class MARAGPipeline(LoggerMixin):
    """
    Multi-Agent RAG Pipeline.
    
    Orchestrates the complete RAG pipeline using multiple specialized agents:
    1. Retriever Agent - Document retrieval
    2. Ranker Agent - Document re-ranking
    3. Summarizer Agent - Answer generation
    4. Supervisor - Coordination and control
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize Multi-Agent RAG Pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.supervisor = None
        self.data_loader = None
        self._initialize_pipeline()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _initialize_pipeline(self) -> None:
        """Initialize the pipeline components."""
        try:
            # Initialize supervisor
            self.supervisor = Supervisor(self.config)
            self.logger.info("Initialized supervisor")
            
            # Initialize data loader
            self.data_loader = DataLoader({"data_dir": "data"})
            self.logger.info("Initialized data loader")
            
            self.logger.info("Pipeline initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def process(self, query: str, **kwargs) -> Any:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: Input query string
            **kwargs: Additional parameters
            
        Returns:
            Any: Pipeline result
        """
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Process through supervisor
            result = self.supervisor.process(query, **kwargs)
            
            if result.success:
                self.logger.info("Query processed successfully")
            else:
                self.logger.error(f"Query processing failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
    
    def load_documents(self, documents) -> None:
        """
        Load documents for retrieval.
        
        Args:
            documents: Either a file path (str) or list of documents to load
        """
        try:
            # If string, load from file
            if isinstance(documents, str):
                import json
                with open(documents, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                self.logger.info(f"Loaded documents from {documents}")
            
            # Get retriever agent from supervisor
            retriever_agent = self.supervisor.agents["retriever"]
            
            # Check if already indexed
            if hasattr(retriever_agent, '_documents_indexed') and retriever_agent._documents_indexed:
                self.logger.info("Documents already indexed, skipping...")
                return
            
            # Index documents in retrievers
            for retriever in retriever_agent.retrievers.values():
                if hasattr(retriever, 'index_documents'):
                    retriever.index_documents(documents)
            
            # Mark as indexed
            retriever_agent._documents_indexed = True
            
            self.logger.info(f"Indexed {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            raise
    
    def load_dataset(self, dataset_name: str) -> None:
        """
        Load a dataset for processing.
        
        Args:
            dataset_name: Name of the dataset to load
        """
        try:
            # Load dataset info
            dataset_info = self.data_loader.load_dataset(dataset_name)
            self.logger.info(f"Loaded dataset: {dataset_info.name}")
            
            # Load documents if available
            if dataset_info.num_documents > 0:
                # This would need to be implemented based on the specific dataset
                self.logger.info(f"Dataset has {dataset_info.num_documents} documents")
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "config_path": self.config_path,
            "supervisor_info": self.supervisor.get_pipeline_info() if self.supervisor else None,
            "data_loader_available": self.data_loader is not None
        }
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents."""
        if self.supervisor:
            return self.supervisor.get_agent_status()
        return {}
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update pipeline configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        try:
            # Update supervisor configuration
            if self.supervisor:
                self.supervisor.update_config(new_config)
            
            # Update data loader configuration
            if self.data_loader:
                self.data_loader.config.update(new_config.get("data_loader", {}))
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline to initial state."""
        try:
            self._initialize_pipeline()
            self.logger.info("Pipeline reset successfully")
            
        except Exception as e:
            self.logger.error(f"Error resetting pipeline: {e}")
            raise
    
    def benchmark(self, test_data: List[Dict[str, Any]], 
                 dataset_name: str = "test") -> Dict[str, Any]:
        """
        Run benchmark on test data.
        
        Args:
            test_data: Test data containing queries and ground truth
            dataset_name: Name of the dataset
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        try:
            from ..evaluation.benchmark import Benchmark
            
            # Initialize benchmark
            benchmark = Benchmark(self.config)
            
            # Run benchmark
            result = benchmark.run_benchmark(self, test_data, dataset_name)
            
            self.logger.info(f"Benchmark completed on {dataset_name} dataset")
            
            return {
                "retrieval_metrics": result.retrieval_metrics,
                "ranking_metrics": result.ranking_metrics,
                "generation_metrics": result.generation_metrics,
                "execution_time": result.execution_time,
                "metadata": result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error running benchmark: {e}")
            raise
