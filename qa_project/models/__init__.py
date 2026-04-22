"""
models/__init__.py
──────────────────────────────────────────────────────────────
Model interfaces for LLM and Retriever
 
IMPORTANT: Always use config.MODE (not MODE) to read current mode
"""
 
# Import config module (not specific values)
import config
 
# Import model classes
from .llm_interface import LLMInterface
from .retriever import Retriever
 
__all__ = ['LLMInterface', 'Retriever']
 