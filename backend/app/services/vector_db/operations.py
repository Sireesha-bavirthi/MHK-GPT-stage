"""
Vector database CRUD operations.
Handles upsert, search, delete operations on Qdrant.
"""

from typing import List, Dict, Any, Optional
import logging
import uuid
from qdrant_client.http import models

from app.services.vector_db.dbstoring import QdrantClient
from app.core.exceptions import VectorDBError

logger = logging.getLogger(__name__)


class VectorDBOperations:
    """Handles CRUD operations on vector database."""
    
    def __init__(self, qdrant_client: QdrantClient):
        """
        Initialize vector DB operations.
        
        Args:
            qdrant_client: Qdrant client instance
        """
        self.client = qdrant_client
        self.collection_name = qdrant_client.collection_name
    
    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Insert or update vectors in the database.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of point IDs (generated if not provided)
            
        Returns:
            List of point IDs
            
        Raises:
            VectorDBError: If upsert operation fails
        """
        if len(texts) != len(embeddings):
            raise VectorDBError("Number of texts and embeddings must match")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # Add text to metadata
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            metadatas[i]["text"] = text
        
        try:
            # Create points
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
                for point_id, embedding, metadata in zip(ids, embeddings, metadatas)
            ]
            
            # Upsert to Qdrant
            self.client.get_client().upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} points to {self.collection_name}")
            return ids
            
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise VectorDBError(f"Failed to upsert vectors: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with text, metadata, and score
        """
        try:
            # Convert filter dict to Qdrant filter
            qdrant_filter = None
            if filter_dict:
                qdrant_filter = self._build_filter(filter_dict)
            
            # Search using query method (new API in qdrant-client 1.16+)
            results = self.client.get_client().query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True
            ).points
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "score": result.score
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorDBError(f"Failed to search vectors: {str(e)}")
    
    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents containing keywords in text payload.
        Uses scroll to get all documents and filters by keyword matches.
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of search results with text, metadata, and keyword match score
        """
        try:
            # Build filter if provided
            qdrant_filter = None
            if filter_dict:
                qdrant_filter = self._build_filter(filter_dict)
            
            # Scroll through all documents (or a large batch)
            # Note: For production, you might want to use full-text search index
            results, _ = self.client.get_client().scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=1000,  # Get a large batch for keyword matching
                with_payload=True,
                with_vectors=False
            )
            
            # Score documents by keyword matches
            scored_results = []
            for point in results:
                text = point.payload.get("text", "").lower()
                
                # Count keyword matches
                keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
                
                if keyword_matches > 0:
                    # Calculate keyword score (normalized by number of keywords)
                    keyword_score = keyword_matches / len(keywords)
                    
                    scored_results.append({
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                        "score": keyword_score
                    })
            
            # Sort by score and return top_k
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_results[:top_k]
            
            logger.info(f"Keyword search found {len(top_results)} results for keywords: {keywords}")
            return top_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            raise VectorDBError(f"Failed to search by keywords: {str(e)}")
    
    def delete_by_id(self, point_ids: List[str]) -> bool:
        """
        Delete points by ID.
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.get_client().delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            logger.info(f"Deleted {len(point_ids)} points")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise VectorDBError(f"Failed to delete points: {str(e)}")
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Delete points by metadata filter.
        
        Args:
            filter_dict: Metadata filters
            
        Returns:
            True if successful
        """
        try:
            qdrant_filter = self._build_filter(filter_dict)
            self.client.get_client().delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter)
            )
            logger.info(f"Deleted points matching filter: {filter_dict}")
            return True
        except Exception as e:
            logger.error(f"Delete by filter failed: {str(e)}")
            raise VectorDBError(f"Failed to delete by filter: {str(e)}")
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all documents with pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List of documents
        """
        try:
            results = self.client.get_client().scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            documents = []
            for point in results[0]:  # results is a tuple (points, next_offset)
                documents.append({
                    "id": point.id,
                    "metadata": point.payload
                })
            
            return documents
        except Exception as e:
            logger.error(f"Get all documents failed: {str(e)}")
            raise VectorDBError(f"Failed to get documents: {str(e)}")
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> models.Filter:
        """
        Build Qdrant filter from dictionary.
        
        Args:
            filter_dict: Filter dictionary
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        for key, value in filter_dict.items():
            if isinstance(value, list):
                # IN filter
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                # Exact match
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=conditions) if conditions else None