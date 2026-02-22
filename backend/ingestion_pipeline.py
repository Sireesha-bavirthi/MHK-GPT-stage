"""
__Title__ : MHK chatbot
__Author_name__: Bhavani Kishore
__Verison__: 1.0
__Created_Date__: 06 february 2026
__Updated_date__: 06 february 2026
__Employee_id: 800341
__Description__ : Data Ingestion Pipeline Orchestrator.

This pipeline combines three main stages:
1. Document Processing - Collects and processes documents (run_pipeline.py logic)
2. Embedding Generation - Generates embeddings for chunks (embed_pipeline.py logic)
3. Vector DB Storage - Stores embeddings in Qdrant (store_to_qdrant.py logic)

"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.document_processor.pipeline import DocumentProcessingPipeline
from app.services.rag.embeddings import (
    EmbeddingService,
    get_paths,
    get_pending_files,
    process_document_file,
    ensure_directories,
)
from app.services.vector_db.dbstoring import QdrantClient
from app.services.vector_db.operations import VectorDBOperations
from app.core.config import settings
from app.core.logging import setup_logging


# =============================================================================
# Data Classes for Pipeline Results
# =============================================================================
@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    success: bool
    duration: float  # seconds
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "duration": self.duration,
            "data": self.data,
            "error": self.error
        }


@dataclass
class PipelineResult:
    """Result from the complete pipeline execution."""
    success: bool
    total_duration: float  # seconds
    stages: List[StageResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "total_duration": self.total_duration,
            "stages": [stage.to_dict() for stage in self.stages],
            "summary": self.summary
        }


# =============================================================================
# Pipeline Orchestrator Class
# =============================================================================
class IngestionPipeline:
    """
    Orchestrates the complete data ingestion pipeline.
    
    This class coordinates three main stages:
    1. Document Processing
    2. Embedding Generation
    3. Vector Database Storage
    """
    
    def __init__(
        self,
        use_semantic_chunking: bool = True,
        similarity_threshold: float = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
        embedding_batch_size: int = 100,
        recreate_collection: bool = False
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            use_semantic_chunking: Whether to use semantic chunking
            similarity_threshold: Threshold for semantic similarity
            min_chunk_size: Minimum chunk size for semantic chunking
            max_chunk_size: Maximum chunk size for semantic chunking
            embedding_batch_size: Batch size for embedding generation
            recreate_collection: Whether to recreate the Qdrant collection
        """
        self.use_semantic_chunking = use_semantic_chunking
        self.similarity_threshold = similarity_threshold or settings.SEMANTIC_SIMILARITY_THRESHOLD
        self.min_chunk_size = min_chunk_size or settings.SEMANTIC_MIN_CHUNK_SIZE
        self.max_chunk_size = max_chunk_size or settings.SEMANTIC_MAX_CHUNK_SIZE
        self.embedding_batch_size = embedding_batch_size
        self.recreate_collection = recreate_collection
        
        # Initialize logger with file handler
        setup_logging()
        self.logger = self._setup_file_logger()
        
        # Pipeline state
        self.stages_completed: List[StageResult] = []
        self.pipeline_start_time: Optional[datetime] = None
    
    def _setup_file_logger(self) -> logging.Logger:
        """
        Setup file-based logger for ingestion pipeline.
        
        Creates a dedicated log file in data/logs/backend/yyyy/mm/dd/ with:
        - Date-based directory structure (data/logs/backend/2026/02/06/)
        - Rotating file handler (max 10MB, 5 backups)
        - Detailed formatting with timestamps and log levels
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        
        # Create logs directory with yyyy/mm/dd structure
        now = datetime.now()
        log_dir = settings.project_root / "data" / "logs" / "backend" / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = now.strftime("%H%M%S")
        log_file = log_dir / f"ingestion_pipeline_{timestamp}.log"
        
        # Create rotating file handler (max 10MB, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter with detailed information
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.DEBUG)
        logger.info(f"Ingestion pipeline logger initialized. Log file: {log_file}")
        
        return logger
    
    # =========================================================================
    # Stage 1: Document Processing
    # =========================================================================
    def _stage_1_process_documents(self) -> StageResult:
        """
        Stage 1: Process raw documents and create chunks.
        
        Returns:
            StageResult with processing statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: DOCUMENT PROCESSING")
        self.logger.info("=" * 80)
        
        stage_start = datetime.now()
        
        try:
            # Create the document processing pipeline
            pipeline = DocumentProcessingPipeline(
                use_semantic_chunking=self.use_semantic_chunking,
                similarity_threshold=self.similarity_threshold,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=self.max_chunk_size,
            )
            
            self.logger.info(f"Configuration:")
            self.logger.info(f"  - Semantic Chunking: {self.use_semantic_chunking}")
            self.logger.info(f"  - Similarity Threshold: {self.similarity_threshold}")
            self.logger.info(f"  - Min Chunk Size: {self.min_chunk_size}")
            self.logger.info(f"  - Max Chunk Size: {self.max_chunk_size}")
            
            # Run the complete pipeline
            result = pipeline.run_complete_pipeline(save_output=True)
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            summary = result['summary']
            self.logger.info(f"[SUCCESS] Stage 1 completed in {stage_duration:.2f}s")
            self.logger.info(f"  - Files processed: {summary['successful']}")
            self.logger.info(f"  - Total chunks: {summary['total_chunks']}")
            
            return StageResult(
                stage_name="Document Processing",
                success=True,
                duration=stage_duration,
                data={
                    "total_files": summary['total_files'],
                    "successful": summary['successful'],
                    "failed": summary['failed'],
                    "total_chunks": summary['total_chunks'],
                    "processed_files": summary.get('processed_files', [])
                }
            )
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"[FAILED] Stage 1 failed: {str(e)}", exc_info=True)
            
            return StageResult(
                stage_name="Document Processing",
                success=False,
                duration=stage_duration,
                error=str(e)
            )
    
    # =========================================================================
    # Stage 2: Embedding Generation
    # =========================================================================
    def _stage_2_generate_embeddings(self) -> StageResult:
        """
        Stage 2: Generate embeddings for document chunks.
        
        Returns:
            StageResult with embedding statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: EMBEDDING GENERATION")
        self.logger.info("=" * 80)
        
        stage_start = datetime.now()
        
        try:
            # Ensure directories exist
            ensure_directories()
            
            paths = get_paths()
            self.logger.info(f"Input directory: {paths['input']}")
            self.logger.info(f"Output directory: {paths['output']}")
            self.logger.info(f"Embedding model: {settings.OPENAI_EMBEDDING_MODEL}")
            
            # Get pending files
            pending_files = get_pending_files()
            
            if not pending_files:
                self.logger.info("No new files to process")
                stage_duration = (datetime.now() - stage_start).total_seconds()
                
                return StageResult(
                    stage_name="Embedding Generation",
                    success=True,
                    duration=stage_duration,
                    data={
                        "files_processed": 0,
                        "files_skipped": 0,
                        "total_chunks": 0
                    }
                )
            
            # Initialize embedding service
            self.logger.info(f"Initializing embedding service (batch size: {self.embedding_batch_size})...")
            service = EmbeddingService(batch_size=self.embedding_batch_size)
            
            # Process files
            total_chunks = 0
            files_processed = 0
            files_skipped = 0
            
            for file_path in pending_files:
                try:
                    chunk_count, output_path = process_document_file(file_path, service)
                    if chunk_count > 0:
                        total_chunks += chunk_count
                        files_processed += 1
                        self.logger.info(f"  [SUCCESS] Processed {file_path.name}: {chunk_count} chunks")
                    else:
                        files_skipped += 1
                        self.logger.warning(f"  [SKIPPED] {file_path.name}: no chunks")
                except Exception as e:
                    self.logger.error(f"  [ERROR] Failed to process {file_path.name}: {e}")
                    files_skipped += 1
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            self.logger.info(f"[SUCCESS] Stage 2 completed in {stage_duration:.2f}s")
            self.logger.info(f"  - Files processed: {files_processed}")
            self.logger.info(f"  - Total chunks embedded: {total_chunks}")
            
            return StageResult(
                stage_name="Embedding Generation",
                success=True,
                duration=stage_duration,
                data={
                    "files_processed": files_processed,
                    "files_skipped": files_skipped,
                    "total_chunks": total_chunks
                }
            )
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"[FAILED] Stage 2 failed: {str(e)}", exc_info=True)
            
            return StageResult(
                stage_name="Embedding Generation",
                success=False,
                duration=stage_duration,
                error=str(e)
            )
    
    # =========================================================================
    # Stage 3: Vector Database Storage
    # =========================================================================
    def _stage_3_store_to_vectordb(self) -> StageResult:
        """
        Stage 3: Store embeddings in Qdrant vector database.
        
        Returns:
            StageResult with storage statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: VECTOR DATABASE STORAGE")
        self.logger.info("=" * 80)
        
        stage_start = datetime.now()
        
        try:
            # Path to embeddings folder
            embeddings_dir = settings.project_root / "data" / "documents" / "embeddings" / "results"
            
            if not embeddings_dir.exists():
                raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
            
            # Find all JSON files
            json_files = list(embeddings_dir.glob("*.json"))
            
            if not json_files:
                self.logger.info("No embedding files found")
                stage_duration = (datetime.now() - stage_start).total_seconds()
                
                return StageResult(
                    stage_name="Vector DB Storage",
                    success=True,
                    duration=stage_duration,
                    data={
                        "new_embeddings_added": 0,
                        "existing_embeddings_skipped": 0,
                        "total_points_in_db": 0
                    }
                )
            
            self.logger.info(f"Found {len(json_files)} embedding files")
            
            # Load all embeddings
            all_documents = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    documents = json.load(f)
                    all_documents.extend(documents)
            
            self.logger.info(f"Loaded {len(all_documents)} total documents")
            
            # Prepare data
            texts = []
            embeddings = []
            metadatas = []
            
            for doc in all_documents:
                if doc.get("page_content") and doc.get("embedding"):
                    texts.append(doc["page_content"])
                    embeddings.append(doc["embedding"])
                    metadatas.append(doc.get("metadata", {}))
            
            self.logger.info(f"Prepared {len(texts)} embeddings for storage")
            
            # Connect to Qdrant
            self.logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}...")
            client = QdrantClient()
            
            # Create collection
            self.logger.info(f"Setting up collection: {settings.QDRANT_COLLECTION_NAME}")
            client.create_collection(recreate=self.recreate_collection)
            
            # Check existing embeddings
            ops = VectorDBOperations(client)
            existing_files = {}
            
            try:
                existing_docs = ops.get_all_documents(limit=1000)
                for doc in existing_docs:
                    file_name = doc['metadata'].get('file_name', '')
                    if file_name:
                        existing_files[file_name] = existing_files.get(file_name, 0) + 1
                
                self.logger.info(f"Found {len(existing_files)} files already in database")
            except:
                self.logger.info("No existing embeddings found (new database)")
            
            # Filter out existing embeddings
            new_texts = []
            new_embeddings = []
            new_metadatas = []
            skipped_count = 0
            
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                file_name = metadata.get('file_name', '')
                
                if file_name in existing_files:
                    skipped_count += 1
                    continue
                
                new_texts.append(text)
                new_embeddings.append(embedding)
                new_metadatas.append(metadata)
            
            if not new_texts:
                self.logger.info(f"All embeddings already exist in database (skipped: {skipped_count})")
                info = client.get_collection_info()
                stage_duration = (datetime.now() - stage_start).total_seconds()
                
                return StageResult(
                    stage_name="Vector DB Storage",
                    success=True,
                    duration=stage_duration,
                    data={
                        "new_embeddings_added": 0,
                        "existing_embeddings_skipped": skipped_count,
                        "total_points_in_db": info['points_count']
                    }
                )
            
            self.logger.info(f"Storing {len(new_texts)} new embeddings...")
            
            # Store new embeddings
            point_ids = ops.upsert(
                texts=new_texts,
                embeddings=new_embeddings,
                metadatas=new_metadatas
            )
            
            # Get final stats
            info = client.get_collection_info()
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            self.logger.info(f"[SUCCESS] Stage 3 completed in {stage_duration:.2f}s")
            self.logger.info(f"  - New embeddings added: {len(point_ids)}")
            self.logger.info(f"  - Existing embeddings skipped: {skipped_count}")
            self.logger.info(f"  - Total points in DB: {info['points_count']}")
            
            return StageResult(
                stage_name="Vector DB Storage",
                success=True,
                duration=stage_duration,
                data={
                    "new_embeddings_added": len(point_ids),
                    "existing_embeddings_skipped": skipped_count,
                    "total_points_in_db": info['points_count'],
                    "collection_name": info['name'],
                    "vector_size": info['vector_size']
                }
            )
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"[FAILED] Stage 3 failed: {str(e)}", exc_info=True)
            
            return StageResult(
                stage_name="Vector DB Storage",
                success=False,
                duration=stage_duration,
                error=str(e)
            )
    
    # =========================================================================
    # Pipeline Execution
    # =========================================================================
    def run(self, skip_stages: Optional[List[int]] = None) -> PipelineResult:
        """
        Run the complete ingestion pipeline.
        
        Args:
            skip_stages: Optional list of stage numbers to skip (1, 2, or 3)
        
        Returns:
            PipelineResult with complete execution details
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DATA INGESTION PIPELINE - START")
        self.logger.info("=" * 80)
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)
        
        self.pipeline_start_time = datetime.now()
        skip_stages = skip_stages or []
        
        # Stage 1: Document Processing
        if 1 not in skip_stages:
            stage1_result = self._stage_1_process_documents()
            self.stages_completed.append(stage1_result)
            
            if not stage1_result.success:
                return self._create_pipeline_result(success=False)
        else:
            self.logger.info("Skipping Stage 1: Document Processing")
        
        # Stage 2: Embedding Generation
        if 2 not in skip_stages:
            stage2_result = self._stage_2_generate_embeddings()
            self.stages_completed.append(stage2_result)
            
            if not stage2_result.success:
                return self._create_pipeline_result(success=False)
        else:
            self.logger.info("Skipping Stage 2: Embedding Generation")
        
        # Stage 3: Vector Database Storage
        if 3 not in skip_stages:
            stage3_result = self._stage_3_store_to_vectordb()
            self.stages_completed.append(stage3_result)
            
            if not stage3_result.success:
                return self._create_pipeline_result(success=False)
        else:
            self.logger.info("Skipping Stage 3: Vector DB Storage")
        
        return self._create_pipeline_result(success=True)
    
    def _create_pipeline_result(self, success: bool) -> PipelineResult:
        """
        Create a pipeline result with summary.
        
        Args:
            success: Whether the pipeline completed successfully
        
        Returns:
            PipelineResult with complete details
        """
        total_duration = (datetime.now() - self.pipeline_start_time).total_seconds()
        
        # Aggregate summary data
        summary = {
            "total_files_processed": 0,
            "total_chunks_created": 0,
            "total_embeddings_generated": 0,
            "embeddings_added_to_db": 0,
            "embeddings_skipped": 0,
            "total_points_in_db": 0
        }
        
        for stage in self.stages_completed:
            if stage.stage_name == "Document Processing" and stage.success:
                summary["total_files_processed"] = stage.data.get("successful", 0)
                summary["total_chunks_created"] = stage.data.get("total_chunks", 0)
            elif stage.stage_name == "Embedding Generation" and stage.success:
                summary["total_embeddings_generated"] = stage.data.get("total_chunks", 0)
            elif stage.stage_name == "Vector DB Storage" and stage.success:
                summary["embeddings_added_to_db"] = stage.data.get("new_embeddings_added", 0)
                summary["embeddings_skipped"] = stage.data.get("existing_embeddings_skipped", 0)
                summary["total_points_in_db"] = stage.data.get("total_points_in_db", 0)
        
        result = PipelineResult(
            success=success,
            total_duration=total_duration,
            stages=self.stages_completed,
            summary=summary
        )
        
        # Log summary
        self.logger.info("\n" + "=" * 80)
        if success:
            self.logger.info("DATA INGESTION PIPELINE - [COMPLETED SUCCESSFULLY]")
        else:
            self.logger.info("DATA INGESTION PIPELINE - [FAILED]")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Duration: {total_duration:.2f}s")
        self.logger.info(f"Stages Completed: {len(self.stages_completed)}")
        self.logger.info("\nSummary:")
        for key, value in summary.items():
            self.logger.info(f"  - {key.replace('_', ' ').title()}: {value}")
        self.logger.info("=" * 80 + "\n")
        
        return result


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point for the ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data Ingestion Pipeline - End-to-end data processing to vector DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python3 ingestion_pipeline.py
  
  # Skip document processing (if already done)
  python3 ingestion_pipeline.py --skip-stages 1
  
  # Skip embedding generation (if already done)
  python3 ingestion_pipeline.py --skip-stages 2
  
  # Run only embedding and storage (skip document processing)
  python3 ingestion_pipeline.py --skip-stages 1
  
  # Recreate vector DB collection
  python3 ingestion_pipeline.py --recreate-collection
        """
    )
    
    parser.add_argument(
        '--skip-stages',
        type=int,
        nargs='+',
        choices=[1, 2, 3],
        help='Stage numbers to skip (1=Document Processing, 2=Embedding Generation, 3=Vector DB Storage)'
    )
    
    parser.add_argument(
        '--no-semantic-chunking',
        action='store_true',
        help='Disable semantic chunking (use fixed-size chunking)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for embedding generation (default: 100)'
    )
    
    parser.add_argument(
        '--recreate-collection',
        action='store_true',
        help='Recreate the Qdrant collection (WARNING: deletes existing data)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create pipeline
        pipeline = IngestionPipeline(
            use_semantic_chunking=not args.no_semantic_chunking,
            embedding_batch_size=args.batch_size,
            recreate_collection=args.recreate_collection
        )
        
        # Run pipeline
        result = pipeline.run(skip_stages=args.skip_stages)
        
        # Print results
        print("\n" + "=" * 80)
        if result.success:
            print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
        else:
            print("[FAILED] PIPELINE FAILED")
        print("=" * 80)
        print(f"\nSUMMARY:")
        print(f"   - Total Duration: {result.total_duration:.2f}s")
        print(f"   - Files Processed: {result.summary['total_files_processed']}")
        print(f"   - Chunks Created: {result.summary['total_chunks_created']}")
        print(f"   - Embeddings Generated: {result.summary['total_embeddings_generated']}")
        print(f"   - Embeddings Added to DB: {result.summary['embeddings_added_to_db']}")
        print(f"   - Total Points in DB: {result.summary['total_points_in_db']}")
        print("\n" + "=" * 80 + "\n")
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Pipeline interrupted by user\n")
        return 1
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n[ERROR] {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
