"""
Qdrant client wrapper for vector database operations.
Handles connection, collection management, and health checks.
"""

from typing import Optional
import logging
from qdrant_client import QdrantClient as QdrantClientSDK
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.exceptions import VectorDBError

logger = logging.getLogger(__name__)


class QdrantClient:
    """Wrapper for Qdrant vector database client."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize Qdrant client.
        Args:
            host: Qdrant host (default from settings)
            port: Qdrant port (default from settings)
            collection_name: Collection name 
        """
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        
        try:
            # Check if cloud configuration is provided
            if settings.QDRANT_URL and settings.QDRANT_API_KEY:
                # Use Qdrant Cloud
                self.client = QdrantClientSDK(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )
                logger.info(f"Connected to Qdrant Cloud at {settings.QDRANT_URL}")
            else:
                # Use local Qdrant (Docker or localhost)
                self.host = host or settings.QDRANT_HOST
                self.port = port or settings.QDRANT_PORT
                self.client = QdrantClientSDK(host=self.host, port=self.port)
                logger.info(f"Connected to local Qdrant at {self.host}:{self.port}")
        except Exception as e:
            raise VectorDBError(f"Failed to connect to Qdrant: {str(e)}")
    
    def create_collection(self, recreate: bool = False):
        """
        Create collection if it doesn't exist.
        
        Args:
            recreate: If True, delete existing collection and recreate
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists and recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name} (dim={self.vector_size})")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Successfully created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            raise VectorDBError(f"Failed to create collection: {str(e)}")
    
    def collection_exists(self) -> bool:
        """
        Check if collection exists.
        
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {str(e)}")
            return False
    
    def get_collection_info(self) -> dict:
        """
        Get collection information.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": self.vector_size,
            }
        except Exception as e:
            raise VectorDBError(f"Failed to get collection info: {str(e)}")
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            if self.collection_exists():
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            else:
                logger.warning(f"Collection does not exist: {self.collection_name}")
        except Exception as e:
            raise VectorDBError(f"Failed to delete collection: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False
    
    def get_client(self) -> QdrantClientSDK:
        """
        Get underlying Qdrant client.
        
        Returns:
            QdrantClient SDK instance
        """
        return self.client
