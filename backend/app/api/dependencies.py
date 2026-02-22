from functools import lru_cache
from fastapi import Depends
from rag_pipeline import RAGPipeline
from ingestion_pipeline import IngestionPipeline

@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create RAG pipeline instance.
    Cached to reuse connections.
    """
    return RAGPipeline()

def get_ingestion_pipeline() -> IngestionPipeline:
    """
    Get ingestion pipeline instance.
    """
    return IngestionPipeline()
