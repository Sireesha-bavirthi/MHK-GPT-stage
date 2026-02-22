"""
__Title__ : MHK chatbot
__Author_name__: Chandolu Jyothirmai
__Description__ : This file imports all functions from embeddings.py and executes them.
__Verison__: 1.0
__Created_Date__: 04 february 2026
__Updated_date__: 05 february 2026
__Employee_id: 800342
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag.embeddings import (
    EmbeddingService,
    get_paths,
    get_pending_files,
    process_document_file,
    ensure_directories,
)
from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Header
    paths = get_paths()
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION PIPELINE")
    print("=" * 80)
    print(f"Input:  {paths['input']}")
    print(f"Output: {paths['output']}")
    print(f"Model:  {settings.OPENAI_EMBEDDING_MODEL}")
    print("=" * 80)

    logger.info("=" * 80)
    logger.info("EMBEDDING GENERATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input directory: {paths['input']}")
    logger.info(f"Output directory: {paths['output']}")
    logger.info(f"Embedding model: {settings.OPENAI_EMBEDDING_MODEL}")
    logger.info("=" * 80)

    # Ensure directories exist
    ensure_directories()

    # Get pending files
    pending_files = get_pending_files()

    if not pending_files:
        print("\n" + "=" * 80)
        print("NO FILES TO PROCESS")
        print("=" * 80)
        print(f"\nNo new files found in: {paths['input']}")
        print("Add JSON files with chunks to process them.")
        print("=" * 80 + "\n")
        logger.info("No pending files to process")
        sys.exit(0)

    # Initialize service
    logger.info("Initializing embedding service...")
    service = EmbeddingService(batch_size=100)

    # Process files
    start_time = datetime.now()
    total_chunks = 0
    files_processed = 0
    files_skipped = 0

    for file_path in pending_files:
        try:
            chunk_count, output_path = process_document_file(file_path, service)
            if chunk_count > 0:
                total_chunks += chunk_count
                files_processed += 1
            else:
                files_skipped += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            files_skipped += 1

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Log summary
    logger.info("=" * 80)
    logger.info("EMBEDDING PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Files processed: {files_processed}")
    logger.info(f"Files skipped: {files_skipped}")
    logger.info(f"Total chunks embedded: {total_chunks}")
    logger.info(f"Processing time: {duration:.2f}s")
    logger.info("=" * 80)

    # Print summary
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nSUMMARY:")
    print(f"   • Files processed: {files_processed}")
    print(f"   • Files skipped: {files_skipped}")
    print(f"   • Total chunks embedded: {total_chunks}")
    print(f"   • Processing time: {duration:.2f}s")
    print(f"\nEmbeddings saved to: {paths['output']}")
    print("=" * 80 + "\n")
    sys.exit(0)