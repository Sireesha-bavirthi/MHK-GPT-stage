"""
__Title__ : MHK chatbot
__Author_name__: Shalini Tata
__Verison__: 1.0
__Created_Date__: 05 february 2026
__Updated_date__: 06 february 2026
__Employee_id: 800338
__Description__ : Retrieval service for RAG system.
Handles document retrieval with multiple strategies (basic similarity, MMR).
"""

import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.core.exceptions import EmbeddingError, VectorDBError
from app.services.rag.embeddings import get_embedding_service
from app.services.vector_db.dbstoring import QdrantClient
from app.services.vector_db.operations import VectorDBOperations

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class RetrievalResult:
    """Container for a single retrieval result."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    preview: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
            "preview": self.preview
        }


# =============================================================================
# Retriever Service
# =============================================================================
class RetrieverService:
    """
    Document retrieval service with multiple search strategies.
    
    Features:
    - Basic similarity search
    - MMR (Maximal Marginal Relevance) for diversity
    - Metadata filtering
    - Soft deduplication (max 2 chunks per document)
    - Configurable score thresholds
    """
    
    # Default configuration
    DEFAULT_TOP_K = 5
    DEFAULT_FETCH_K = 20
    DEFAULT_LAMBDA_MULT = 0.7
    DEFAULT_SCORE_THRESHOLD = 0.2  # Lowered to retrieve founding date chunk
    DEFAULT_PREVIEW_LENGTH = 200
    MAX_CHUNKS_PER_DOC = 3  # Increased to include founding date chunk
    
    def __init__(
        self,
        embedding_service=None,
        vector_db_operations=None,
        top_k: int = None,
        fetch_k: int = None,
        lambda_mult: float = None,
        score_threshold: float = None,
        preview_length: int = None
    ):
        """
        Initialize retriever service.
        
        Args:
            embedding_service: EmbeddingService instance (optional)
            vector_db_operations: VectorDBOperations instance (optional)
            top_k: Number of results to return
            fetch_k: Number of candidates for MMR
            lambda_mult: MMR lambda parameter (relevance vs diversity)
            score_threshold: Minimum similarity score
            preview_length: Length of text preview
        """
        # Initialize services
        self.embedding_service = embedding_service or get_embedding_service()
        
        if vector_db_operations is None:
            qdrant_client = QdrantClient()
            self.vector_db_ops = VectorDBOperations(qdrant_client)
        else:
            self.vector_db_ops = vector_db_operations
        
        # Configuration
        self.top_k = top_k or settings.RETRIEVAL_TOP_K or self.DEFAULT_TOP_K
        self.fetch_k = fetch_k or settings.RETRIEVAL_FETCH_K or self.DEFAULT_FETCH_K
        self.lambda_mult = lambda_mult or settings.RETRIEVAL_LAMBDA_MULT or self.DEFAULT_LAMBDA_MULT
        self.score_threshold = score_threshold or self.DEFAULT_SCORE_THRESHOLD
        self.preview_length = preview_length or self.DEFAULT_PREVIEW_LENGTH
        
        # Hybrid search weights
        self.keyword_weight = settings.HYBRID_KEYWORD_WEIGHT
        self.semantic_weight = settings.HYBRID_SEMANTIC_WEIGHT
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "avg_query_time": 0.0,
            "strategy_counts": defaultdict(int)
        }
        
        logger.info(
            f"RetrieverService initialized | top_k={self.top_k} | "
            f"fetch_k={self.fetch_k} | lambda={self.lambda_mult} | "
            f"threshold={self.score_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        strategy: str = "mmr",
        top_k: int = None,
        score_threshold: float = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        full_text: bool = False
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            strategy: Retrieval strategy ("mmr" or "basic")
            top_k: Number of results (overrides default)
            score_threshold: Minimum score (overrides default)
            metadata_filter: Filter by metadata fields
            full_text: Return full text in results (not just preview)
            
        Returns:
            List of RetrievalResult objects
            
        Raises:
            EmbeddingError: If query embedding fails
            VectorDBError: If vector search fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Sanitize query for logging
        sanitized_query = query[:100] + "..." if len(query) > 100 else query
        
        # Start timing
        start_time = time.time()
        
        # Use provided values or defaults
        k = top_k or self.top_k
        threshold = score_threshold or self.score_threshold
        
        logger.info(
            f"Retrieval query: '{sanitized_query}' | "
            f"strategy={strategy} | top_k={k} | threshold={threshold}"
        )
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed(query)
            
            # Execute search based on strategy
            if strategy.lower() == "mmr":
                results = self._mmr_search(
                    query_embedding=query_embedding,
                    top_k=k,
                    score_threshold=threshold,
                    metadata_filter=metadata_filter
                )
            elif strategy.lower() == "basic":
                results = self._basic_search(
                    query_embedding=query_embedding,
                    top_k=k,
                    score_threshold=threshold,
                    metadata_filter=metadata_filter
                )
            elif strategy.lower() == "hybrid":
                results = self._hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=k,
                    score_threshold=threshold,
                    metadata_filter=metadata_filter
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'mmr', 'basic', or 'hybrid'")
            
            # Apply soft deduplication
            results = self._apply_soft_deduplication(results)
            
            # Format results
            formatted_results = self._format_results(results, full_text=full_text)
            
            # Calculate timing
            query_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(strategy, len(formatted_results), query_time)
            
            # Log results
            logger.info(
                f"Retrieved {len(formatted_results)} documents | "
                f"strategy={strategy} | time={query_time:.3f}s | "
                f"scores=[{', '.join(f'{r.score:.3f}' for r in formatted_results[:3])}...]"
            )
            
            # Log document IDs and content preview for debugging
            doc_ids = [r.id for r in formatted_results]
            logger.debug(f"Retrieved document IDs: {doc_ids}")
            
            # Log detailed content for debugging
            for i, r in enumerate(formatted_results[:5], 1):
                logger.info(
                    f"  Doc {i}: score={r.score:.3f} | file={r.metadata.get('file_name', 'unknown')} | "
                    f"content={r.text[:100]}..."
                )

            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def _basic_search(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Basic similarity search using vector database.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum similarity score
            metadata_filter: Metadata filters
            
        Returns:
            List of search results
        """
        logger.debug(f"Executing basic search | top_k={top_k}")
        
        # Search vector database
        results = self.vector_db_ops.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch more for deduplication
            filter_dict=metadata_filter,
            score_threshold=score_threshold
        )
        
        # If no results and threshold is high, try adaptive threshold
        if not results and score_threshold > 0.5:
            logger.warning(f"No results with threshold={score_threshold}, trying adaptive threshold")
            results = self._adaptive_threshold_search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                metadata_filter=metadata_filter
            )
        
        return results
    
    def _mmr_search(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance search for diversity.
        
        MMR balances relevance and diversity:
        MMR = λ × Similarity(query, doc) - (1-λ) × max(Similarity(doc, selected_docs))
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum similarity score
            metadata_filter: Metadata filters
            
        Returns:
            List of search results with diversity
        """
        logger.debug(f"Executing MMR search | top_k={top_k} | fetch_k={self.fetch_k}")
        
        # Fetch more candidates than needed
        fetch_k = max(self.fetch_k, top_k * 3)
        
        candidates = self.vector_db_ops.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filter_dict=metadata_filter,
            score_threshold=score_threshold
        )
        
        # If no candidates and threshold is high, try adaptive threshold
        if not candidates and score_threshold > 0.5:
            logger.warning(f"No candidates with threshold={score_threshold}, trying adaptive threshold")
            candidates = self._adaptive_threshold_search(
                query_embedding=query_embedding,
                top_k=fetch_k,
                metadata_filter=metadata_filter
            )
        
        if not candidates:
            logger.warning("No candidates found for MMR")
            return []
        
        # Apply MMR algorithm
        mmr_results = self._calculate_mmr(
            query_embedding=query_embedding,
            candidates=candidates,
            top_k=top_k * 2  # Get more for deduplication
        )
        
        return mmr_results
    
    def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining keyword and semantic search.
        
        Uses weighted score fusion:
        hybrid_score = (keyword_weight × keyword_score) + (semantic_weight × semantic_score)
        
        Args:
            query: Query string for keyword extraction
            query_embedding: Query embedding vector
            top_k: Number of results
            score_threshold: Minimum similarity score
            metadata_filter: Metadata filters
            
        Returns:
            List of fused search results
        """
        logger.debug(f"Executing hybrid search | top_k={top_k}")
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        logger.info(f"Extracted keywords for hybrid search: {keywords}")
        
        # Perform semantic search
        semantic_results = self.vector_db_ops.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Fetch more candidates
            filter_dict=metadata_filter,
            score_threshold=score_threshold * 0.7  # Lower threshold for more candidates
        )
        
        # Perform keyword search
        keyword_results = self.vector_db_ops.keyword_search(
            keywords=keywords,
            top_k=top_k * 3,  # Fetch more candidates
            filter_dict=metadata_filter
        )
        
        # Fuse results using weighted combination
        fused_results = self._fuse_scores(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight
        )
        
        # Sort by fused score and return top_k
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = fused_results[:top_k * 2]  # Get more for deduplication
        
        logger.info(
            f"Hybrid search completed | semantic={len(semantic_results)} | "
            f"keyword={len(keyword_results)} | fused={len(fused_results)}"
        )
        
        return top_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query text.
        
        Simple approach:
        - Remove common stop words
        - Extract meaningful words (3+ chars)
        - Include named entities and numbers
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        # Common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'about', 'what',
            'when', 'where', 'who', 'how', 'why', 'this', 'that', 'these', 'those'
        }
        
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter keywords
        keywords = [
            word for word in words
            if len(word) >= 3 and word not in stop_words
        ]
        
        # Also include capitalized words and numbers from original query
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        numbers = re.findall(r'\b\d+\b', query)
        
        # Combine all keywords
        all_keywords = keywords + [w.lower() for w in capitalized] + numbers
        
        # Remove duplicates while preserving order
        seen: Set[str] = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _fuse_scores(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Fuse semantic and keyword search results using weighted combination.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores (default 0.7)
            keyword_weight: Weight for keyword scores (default 0.3)
            
        Returns:
            List of fused results with combined scores
        """
        # Create lookup dictionaries
        semantic_scores = {r["id"]: r["score"] for r in semantic_results}
        keyword_scores = {r["id"]: r["score"] for r in keyword_results}
        
        # Get all unique document IDs
        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        # Create result lookup
        result_lookup = {}
        for r in semantic_results:
            result_lookup[r["id"]] = r
        for r in keyword_results:
            if r["id"] not in result_lookup:
                result_lookup[r["id"]] = r
        
        # Normalize scores to [0, 1] range
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1.0
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1.0
        
        # Fuse scores
        fused_results = []
        for doc_id in all_ids:
            # Get normalized scores (0 if not present)
            semantic_score = semantic_scores.get(doc_id, 0) / max_semantic
            keyword_score = keyword_scores.get(doc_id, 0) / max_keyword
            
            # Weighted combination
            fused_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            
            # Get result object
            if doc_id in result_lookup:
                result = result_lookup[doc_id].copy()
                result["score"] = fused_score
                result["semantic_score"] = semantic_score
                result["keyword_score"] = keyword_score
                fused_results.append(result)
        
        logger.debug(
            f"Score fusion completed | total_docs={len(fused_results)} | "
            f"weights=(semantic={semantic_weight}, keyword={keyword_weight})"
        )
        
        return fused_results
    
    def _calculate_mmr(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Calculate Maximal Marginal Relevance using sklearn.
        
        Args:
            query_embedding: Query embedding vector
            candidates: Candidate documents with embeddings
            top_k: Number of results to select
            
        Returns:
            List of selected documents
        """
        if not candidates:
            return []
        
        # Get embeddings from candidates (need to fetch them)
        # For now, we'll use a simplified approach since embeddings aren't in search results
        # We'll select based on score diversity instead
        
        selected = []
        remaining = candidates.copy()
        
        # Select first document (highest score)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Iteratively select documents
        while len(selected) < top_k and remaining:
            best_score = -1
            best_idx = 0
            
            for idx, candidate in enumerate(remaining):
                # Relevance score (from vector search)
                relevance = candidate["score"]
                
                # Diversity penalty (simple heuristic: different documents)
                diversity_penalty = 0
                for selected_doc in selected:
                    # Penalize if from same document
                    if candidate["metadata"].get("file_name") == selected_doc["metadata"].get("file_name"):
                        diversity_penalty += 0.3
                
                # MMR score
                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _adaptive_threshold_search(
        self,
        query_embedding: List[float],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Adaptive threshold search with fallback.
        
        Tries progressively lower thresholds: 0.7 → 0.5 → 0.3 → 0.0
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            metadata_filter: Metadata filters
            
        Returns:
            List of search results
        """
        thresholds = [0.5, 0.3, 0.0]
        
        for threshold in thresholds:
            logger.debug(f"Trying threshold={threshold}")
            results = self.vector_db_ops.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=metadata_filter,
                score_threshold=threshold
            )
            
            if results:
                logger.info(f"Found {len(results)} results with adaptive threshold={threshold}")
                return results
        
        logger.warning("No results found even with threshold=0.0")
        return []
    
    def _apply_soft_deduplication(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply soft deduplication: max 2 chunks per document.
        
        Args:
            results: Search results
            
        Returns:
            Deduplicated results
        """
        doc_counts = defaultdict(int)
        deduplicated = []
        
        for result in results:
            file_name = result["metadata"].get("file_name", "unknown")
            
            if doc_counts[file_name] < self.MAX_CHUNKS_PER_DOC:
                deduplicated.append(result)
                doc_counts[file_name] += 1
        
        if len(deduplicated) < len(results):
            logger.debug(
                f"Soft deduplication: {len(results)} → {len(deduplicated)} results | "
                f"max_chunks_per_doc={self.MAX_CHUNKS_PER_DOC}"
            )
        
        return deduplicated
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        full_text: bool = False
    ) -> List[RetrievalResult]:
        """
        Format search results with previews.
        
        Args:
            results: Raw search results
            full_text: Include full text (not just preview)
            
        Returns:
            List of RetrievalResult objects
        """
        formatted = []
        
        for result in results:
            text = result.get("text", "")
            preview = self._generate_preview(text) if not full_text else text
            
            formatted.append(RetrievalResult(
                id=result.get("id", ""),
                text=text,
                metadata=result.get("metadata", {}),
                score=result.get("score", 0.0),
                preview=preview
            ))
        
        return formatted
    
    def _generate_preview(self, text: str) -> str:
        """
        Generate preview text with smart truncation.
        
        Args:
            text: Full text
            
        Returns:
            Preview text (~200 chars)
        """
        if len(text) <= self.preview_length:
            return text
        
        # Truncate at word boundary
        preview = text[:self.preview_length].rsplit(' ', 1)[0]
        return preview + "..."
    
    def _update_stats(self, strategy: str, result_count: int, query_time: float):
        """Update retrieval statistics."""
        self.stats["total_queries"] += 1
        self.stats["total_results"] += result_count
        self.stats["strategy_counts"][strategy] += 1
        
        # Update average query time
        total_queries = self.stats["total_queries"]
        current_avg = self.stats["avg_query_time"]
        self.stats["avg_query_time"] = (
            (current_avg * (total_queries - 1) + query_time) / total_queries
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_queries": self.stats["total_queries"],
            "total_results": self.stats["total_results"],
            "avg_results_per_query": (
                self.stats["total_results"] / self.stats["total_queries"]
                if self.stats["total_queries"] > 0 else 0
            ),
            "avg_query_time": self.stats["avg_query_time"],
            "strategy_counts": dict(self.stats["strategy_counts"])
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "avg_query_time": 0.0,
            "strategy_counts": defaultdict(int)
        }
        logger.info("Statistics reset")


# =============================================================================
# Singleton Instance
# =============================================================================
_retriever_service: Optional[RetrieverService] = None


def get_retriever_service() -> RetrieverService:
    """Get or create singleton RetrieverService instance."""
    global _retriever_service
    if _retriever_service is None:
        _retriever_service = RetrieverService()
    return _retriever_service