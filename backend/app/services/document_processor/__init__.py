"""
__Project__: Company Chatbot
__Description__: Document Processor Package that provides comprehensive document processing for RAG pipeline. Includes loader, cleaner, chunker, metadata extractor, and pipeline orchestration for PDF, DOCX, TXT, and MD files.
__Created Date__: 04-02-2026
__Updated Date__: 06-02-2026
__Author__: Nagamani Bhukya
__Employee Id__: 800339
"""

# =============================================================================
# PACKAGE EXPORTS
# =============================================================================

from app.services.document_processor.loader import DocumentLoader
from app.services.document_processor.cleaner import TextCleaner
from app.services.document_processor.chunker import Chunker
from app.services.document_processor.metadata_extractor import MetadataExtractor
from app.services.document_processor.pipeline import DocumentProcessingPipeline


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    'DocumentLoader',
    'TextCleaner',
    'Chunker',
    'MetadataExtractor',
    'DocumentProcessingPipeline',
]
