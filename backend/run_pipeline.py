"""
__Project__: Company Chatbot
__Description__: Script to run the document processing pipeline. Processes raw documents through loading, cleaning, chunking, and metadata extraction stages.
__Created Date__: 04-02-2026
__Updated Date__: 06-02-2026
__Author__: Nagamani Bhukya
__Employee Id__: 800339
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.document_processor.pipeline import DocumentProcessingPipeline
from app.core.config import settings


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run the complete document processing pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("DOCUMENT PROCESSING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Raw documents path: {settings.raw_documents_path_absolute}")
        logger.info(f"Processed documents path: {settings.processed_documents_path_absolute}")
        logger.info(f"Embedding model: {settings.OPENAI_EMBEDDING_MODEL}")
        logger.info(f"Semantic similarity threshold: {settings.SEMANTIC_SIMILARITY_THRESHOLD}")
        logger.info("=" * 80)

        pipeline = DocumentProcessingPipeline(
            use_semantic_chunking=True,
            similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
            min_chunk_size=settings.SEMANTIC_MIN_CHUNK_SIZE,
            max_chunk_size=settings.SEMANTIC_MAX_CHUNK_SIZE,
        )

        result = pipeline.run_complete_pipeline(save_output=True)

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        summary = result['summary']
        print("\nPROCESSING SUMMARY:")
        print(f"   Total files processed: {summary['total_files']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Total chunks created: {summary['total_chunks']}")

        if summary.get('processed_files'):
            print(f"\nProcessed files:")
            for f in summary['processed_files']:
                print(f"   - {f}")

        print(f"\nJSON results saved to: {settings.processed_documents_path_absolute}/results/")
        print(f"Original files moved to: {settings.processed_documents_path_absolute}/")

        print("\n" + "=" * 80)
        print("Documents are ready for embedding generation!")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}", exc_info=True)
        print("\n" + "=" * 80)
        print("PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        return 1


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())
