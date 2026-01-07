"""
Logging utilities for Multi-Agent RAG system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(name: str = "ma_rag", 
                level: str = "INFO",
                log_file: Optional[str] = None,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logger for the Multi-Agent RAG system.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ma_rag") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_retrieval_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log retrieval performance metrics."""
        self.logger.info("Retrieval Metrics:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                self.logger.info(f"  {k}:")
                for k2, v2 in v.items():
                    self.logger.info(f"    {k2}: {v2:.4f}")
            else:
                self.logger.info(f"  {k}: {v:.4f}")
    
    def log_ranking_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log ranking performance metrics."""
        self.logger.info("Ranking Metrics:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                self.logger.info(f"  {k}:")
                for k2, v2 in v.items():
                    self.logger.info(f"    {k2}: {v2:.4f}")
            else:
                self.logger.info(f"  {k}: {v:.4f}")
    
    def log_generation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log generation performance metrics."""
        self.logger.info("Generation Metrics:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                self.logger.info(f"  {k}:")
                for k2, v2 in v.items():
                    self.logger.info(f"    {k2}: {v2:.4f}")
            else:
                self.logger.info(f"  {k}: {v:.4f}")
    
    def log_pipeline_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log overall pipeline metrics."""
        self.logger.info("Pipeline Metrics:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v}")
