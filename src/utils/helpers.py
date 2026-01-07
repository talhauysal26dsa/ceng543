"""
Helper utilities for Multi-Agent RAG system.
"""

import re
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    return text.strip()


def generate_doc_id(text: str) -> str:
    """
    Generate a unique document ID from text content.
    
    Args:
        text: Document text
        
    Returns:
        str: Generated document ID
    """
    # Create hash from text content
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f"doc_{text_hash[:8]}"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency-based approach.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List[str]: Extracted keywords
    """
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_document(doc: Dict[str, Any], max_text_length: int = 200) -> str:
    """
    Format document for display.
    
    Args:
        doc: Document dictionary
        max_text_length: Maximum text length to display
        
    Returns:
        str: Formatted document string
    """
    doc_id = doc.get("id", "unknown")
    text = doc.get("text", "")
    
    # Truncate text if too long
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
    
    return f"ID: {doc_id}\nText: {text}"


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to load file from
        
    Returns:
        Any: Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries, with dict2 values taking precedence.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        List[Any]: Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        bool: True if valid, False otherwise
    """
    for key in required_keys:
        if key not in config:
            return False
    return True


def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested value from dictionary using dot notation.
    
    Args:
        data: Dictionary to search
        key_path: Dot-separated key path (e.g., "config.model.name")
        default: Default value if key not found
        
    Returns:
        Any: Value at key path or default
    """
    keys = key_path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested value in dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        key_path: Dot-separated key path (e.g., "config.model.name")
        value: Value to set
    """
    keys = key_path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
    """
    Format metrics dictionary for display.
    
    Args:
        metrics: Metrics dictionary
        precision: Decimal precision
        
    Returns:
        str: Formatted metrics string
    """
    lines = []
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    lines.append(f"  {sub_key}: {sub_value:.{precision}f}")
                else:
                    lines.append(f"  {sub_key}: {sub_value}")
        else:
            if isinstance(value, (int, float)):
                lines.append(f"{key}: {value:.{precision}f}")
            else:
                lines.append(f"{key}: {value}")
    
    return "\n".join(lines)
