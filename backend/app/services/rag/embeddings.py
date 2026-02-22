"""
__Title__ : MHK chatbot
__Author_name__: Chandolu Jyothirmai
__Description__ : This file fetches the chunks created and creates embeddings for chunks ussing OpenAI model and Exponetial backoff for rate limits
__Verison__: 1.0
__Created_Date__: 04 february 2026
__Updated_date__: 05 february 2026
__Employee_id: 800342
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
load_dotenv()
# =============================================================================
# Path Configuration
# =============================================================================
def get_paths() -> Dict[str, Path]:
    """Get all embedding pipeline paths."""
    base = settings.project_root / "data" / "documents"
    return {
        "input": base / "processed" / "results",
        "output": base / "embeddings" / "results",
    }

# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

# =============================================================================
# EmbeddingService Class
# =============================================================================
class EmbeddingService:
    """Service for generating text embeddings using OpenAI. 
    Features batch processing and exponential backoff for rate limits."""

    # Configuration
    DEFAULT_BATCH_SIZE = 100
    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 60.0

    # Initialising the service 
    def __init__(self, model: str = None, batch_size: int = None):
        """Initialize embedding service."""
        self.model_name = model or settings.OPENAI_EMBEDDING_MODEL
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.dimension = 1536
        self._init_client()
        logger.info(
            f"EmbeddingService initialized | model={self.model_name} | "
            f"batch_size={self.batch_size} | dimension={self.dimension}"
        )

    # Initialising the OpenAI client
    def _init_client(self):
        """Initialize the OpenAI embeddings client."""
        try:
            if not settings.OPENAI_API_KEY:
                raise EmbeddingError("OPENAI_API_KEY not configured")
            self.client = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=self.model_name
            )
            logger.info(f"OpenAI embeddings client initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise EmbeddingError(f"Client initialization failed: {e}")

    # Calculate exponential backoff time
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.INITIAL_RETRY_DELAY * (2 ** attempt)
        return min(delay, self.MAX_RETRY_DELAY)
    
    # Embedding with retry
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with exponential backoff retry."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                embeddings = self.client.embed_documents(texts)
                return embeddings
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str or "429" in error_str:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Rate limit hit, retry {attempt + 1}/{self.MAX_RETRIES} "
                        f"after {delay:.1f}s | error={e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Embedding error (non-retryable): {e}")
                    raise EmbeddingError(f"Embedding failed: {e}")
        logger.error(f"All {self.MAX_RETRIES} retries failed: {last_error}")
        raise EmbeddingError(f"Embedding failed after {self.MAX_RETRIES} retries: {last_error}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        try:
            embeddings = self._embed_with_retry([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Single embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        if not texts:
            return []
        valid_pairs = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid_pairs:
            raise EmbeddingError("No valid texts to embed")
        valid_indices, valid_texts = zip(*valid_pairs)
        valid_texts = list(valid_texts)
        logger.info(f"Embedding {len(valid_texts)} texts in batches of {self.batch_size}")
        all_embeddings = []
        total_batches = (len(valid_texts) + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, len(valid_texts), self.batch_size):
            batch_num = batch_idx // self.batch_size + 1
            batch_texts = valid_texts[batch_idx:batch_idx + self.batch_size]
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            try:
                batch_embeddings = self._embed_with_retry(batch_texts)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {batch_num}/{total_batches} completed")
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                raise
        logger.info(f"Embedding complete: {len(all_embeddings)} vectors generated")
        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding vector dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

# =============================================================================
# File Processing Functions
# =============================================================================
def ensure_directories():
    """Create required directories if they don't exist."""
    paths = get_paths()
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")


def get_pending_files() -> List[Path]:
    """Get list of files pending embedding generation."""
    paths = get_paths()
    input_dir = paths["input"]
    output_dir = paths["output"]

    if not input_dir.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        return []
    pending = []
    for file_path in input_dir.glob("*.json"):
        output_file = output_dir / file_path.name
        if not output_file.exists():
            pending.append(file_path)
            logger.info(f"Found pending file: {file_path.name}")
        else:
            logger.debug(f"Skipping already processed: {file_path.name}")
    logger.info(f"Found {len(pending)} pending files")
    return pending


def load_document_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """Load document chunks from JSON file."""
    logger.info(f"Loading chunks from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
    return chunks


def save_embeddings(documents: List[Dict[str, Any]], output_path: Path):
    """Save documents with embeddings to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(documents)} documents to: {output_path}")


def process_document_file(file_path: Path, service: EmbeddingService) -> Tuple[int, Path]:
    """ Process a single document file: generate embeddings and save. """
    paths = get_paths()
    logger.info(f"Processing file: {file_path.name}")

    # Load chunks
    chunks = load_document_chunks(file_path)
    if not chunks:
        logger.warning(f"No chunks found in {file_path.name}")
        return 0, None

    # Extract texts
    texts = [chunk.get("page_content", "") for chunk in chunks]

    # Generate embeddings
    embeddings = service.embed_batch(texts)

    # Add embeddings to documents
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["metadata"]["embedding_model"] = service.get_model_name()
        chunk["metadata"]["embedding_dimension"] = service.get_dimension()
        chunk["metadata"]["embedded_at"] = datetime.now().isoformat()

    # Save to results directory
    output_path = paths["output"] / file_path.name
    save_embeddings(chunks, output_path)
    logger.info(f"Completed {file_path.name}: {len(chunks)} chunks embedded")
    return len(chunks), output_path


# =============================================================================
# Singleton Instance
# =============================================================================
_embedding_service: Optional[EmbeddingService] = None
def get_embedding_service() -> EmbeddingService:
    """Get or create singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service