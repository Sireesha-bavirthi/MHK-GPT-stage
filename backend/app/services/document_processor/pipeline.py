"""
__Project__: Company Chatbot
__Description__: Document Processing Pipeline Module that orchestrates the complete workflow from loading to embedding preparation. Handles batch and incremental processing with automatic file organization and JSON output generation.
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
import json
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.services.document_processor.loader import DocumentLoader
from app.services.document_processor.cleaner import TextCleaner
from app.services.document_processor.chunker import Chunker
from app.services.document_processor.metadata_extractor import MetadataExtractor


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Supported file extensions for processing
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md']


# =============================================================================
# DOCUMENT PROCESSING PIPELINE CLASS
# =============================================================================

class DocumentProcessingPipeline:
    """
    Document Processing Pipeline Class.

    End-to-end document processing pipeline that coordinates loading, cleaning,
    chunking, and metadata extraction. Supports both batch and incremental
    processing with automatic file organization.

    Pipeline stages:
        1. Load documents (using LangChain loaders)
        2. Clean text (normalize and remove noise)
        3. Semantic chunking (group by semantic similarity)
        4. Extract metadata (enrich with statistics and info)
        5. Return documents ready for embedding

    Attributes:
        use_semantic_chunking (bool): Whether semantic chunking is enabled.
        loader (DocumentLoader): Document loader instance.
        cleaner (TextCleaner): Text cleaner instance.
        chunker (Chunker): Document chunker instance.
        metadata_extractor (MetadataExtractor): Metadata extractor instance.

    Example:
        >>> pipeline = DocumentProcessingPipeline()
        >>> result = pipeline.run_complete_pipeline()
        >>> print(f"Processed {result['summary']['total_files']} files")
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(
        self,
        use_semantic_chunking: bool = True,
        similarity_threshold: float = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
    ):
        """
        Initialize the document processing pipeline.

        Args:
            use_semantic_chunking (bool): Whether to use semantic chunking (default: True).
            similarity_threshold (float): Threshold for semantic similarity (optional).
            min_chunk_size (int): Minimum chunk size in characters (optional).
            max_chunk_size (int): Maximum chunk size in characters (optional).

        Note:
            If chunking parameters are None, defaults from config.py are used.
        """
        self.use_semantic_chunking = use_semantic_chunking

        # ----- INITIALIZE PIPELINE COMPONENTS -----
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()

        # ----- INITIALIZE CHUNKER (if enabled) -----
        if self.use_semantic_chunking:
            self.chunker = Chunker(
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
            )
        else:
            self.chunker = None

        logger.info("Document processing pipeline initialized")

    # -------------------------------------------------------------------------
    # CORE DOCUMENT PROCESSING
    # -------------------------------------------------------------------------

    def process_documents(
        self,
        documents: List[Document],
        skip_cleaning: bool = False,
        skip_chunking: bool = False,
        skip_metadata: bool = False,
    ) -> List[Document]:
        """
        Process a list of documents through the complete pipeline.

        Applies cleaning, chunking, and metadata extraction in sequence.
        Each stage can be optionally skipped.

        Args:
            documents (List[Document]): List of LangChain Document objects.
            skip_cleaning (bool): Skip text cleaning step (default: False).
            skip_chunking (bool): Skip chunking step (default: False).
            skip_metadata (bool): Skip metadata extraction step (default: False).

        Returns:
            List[Document]: List of processed Document objects ready for embedding.

        Example:
            >>> processed = pipeline.process_documents(raw_docs)
            >>> print(f"Created {len(processed)} chunks")
        """
        # Handle empty input
        if not documents:
            logger.warning("No documents provided for processing")
            return []

        processed_docs = documents
        logger.info(f"Starting pipeline with {len(documents)} documents")

        # ----- STAGE 1: Clean text -----
        if not skip_cleaning:
            logger.info("Stage 1/3: Cleaning text...")
            processed_docs = self.cleaner.clean_and_filter_documents(processed_docs)
            logger.info(f"After cleaning: {len(processed_docs)} documents")

        # ----- STAGE 2: Chunk documents -----
        if not skip_chunking and self.use_semantic_chunking and self.chunker:
            logger.info("Stage 2/3: Semantic chunking...")
            processed_docs = self.chunker.chunk_documents(processed_docs)
            logger.info(f"After chunking: {len(processed_docs)} chunks")

        # ----- STAGE 3: Extract and enrich metadata -----
        if not skip_metadata:
            logger.info("Stage 3/3: Extracting metadata...")
            processed_docs = self.metadata_extractor.enrich_documents_metadata(processed_docs)
            logger.info(f"After metadata extraction: {len(processed_docs)} documents")

        logger.info(f"Pipeline completed. Final document count: {len(processed_docs)}")
        return processed_docs

    # -------------------------------------------------------------------------
    # SINGLE FILE PROCESSING
    # -------------------------------------------------------------------------

    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a single file through the complete pipeline.

        Loads the file and processes it through all pipeline stages.

        Args:
            file_path (str): Path to the document file.

        Returns:
            List[Document]: List of processed Document objects.

        Example:
            >>> docs = pipeline.process_file("/data/report.pdf")
            >>> print(f"Created {len(docs)} chunks from report.pdf")
        """
        logger.info(f"Processing file: {file_path}")

        # Load document
        documents = self.loader.load_document(file_path)

        # Process through pipeline
        processed_docs = self.process_documents(documents)

        return processed_docs

    # -------------------------------------------------------------------------
    # JSON OUTPUT SAVING
    # -------------------------------------------------------------------------

    def save_file_json(self, documents: List[Document], file_name: str) -> str:
        """
        Save processed documents for a single file to its own JSON file.

        Creates a JSON file in the results directory with all chunks
        from the processed document.

        Args:
            documents (List[Document]): List of processed Document objects.
            file_name (str): Original file name (e.g., 'report.pdf').

        Returns:
            str: Path to the saved JSON file.

        Example:
            >>> json_path = pipeline.save_file_json(chunks, "report.pdf")
            >>> print(f"Saved to: {json_path}")  # processed/results/report.json
        """
        # ----- CREATE RESULTS DIRECTORY -----
        results_dir = Path(settings.processed_documents_path_absolute) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # ----- CREATE JSON FILENAME -----
        # Convert "report.pdf" to "report.json"
        json_filename = Path(file_name).stem + ".json"
        output_file = results_dir / json_filename

        # ----- CONVERT DOCUMENTS TO SERIALIZABLE FORMAT -----
        docs_data = []
        for doc in documents:
            docs_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })

        # ----- SAVE TO JSON -----
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(documents)} chunks to: {output_file}")
        return str(output_file)

    # -------------------------------------------------------------------------
    # FILE ORGANIZATION
    # -------------------------------------------------------------------------

    def move_file_to_processed(self, source_path: str) -> str:
        """
        Move a processed file from raw to processed directory.

        Preserves the folder structure when moving files.
        For example: raw/pdf/report.pdf -> processed/pdf/report.pdf

        Args:
            source_path (str): Path to the source file in raw directory.

        Returns:
            str: Path to the moved file in processed directory.

        Example:
            >>> new_path = pipeline.move_file_to_processed("/data/raw/pdf/report.pdf")
            >>> print(new_path)  # /data/processed/pdf/report.pdf
        """
        source = Path(source_path)
        raw_base = Path(settings.raw_documents_path_absolute)
        processed_base = Path(settings.processed_documents_path_absolute)

        # ----- GET RELATIVE PATH FROM RAW DIRECTORY -----
        try:
            relative_path = source.relative_to(raw_base)
        except ValueError:
            # If file is not under raw_base, just use the filename
            relative_path = Path(source.name)

        # ----- CREATE DESTINATION PATH (maintain folder structure) -----
        dest_path = processed_base / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # ----- MOVE THE FILE -----
        shutil.move(str(source), str(dest_path))
        logger.info(f"Moved file: {source.name} -> {dest_path}")

        return str(dest_path)

    # -------------------------------------------------------------------------
    # PROCESS AND MOVE (SINGLE FILE)
    # -------------------------------------------------------------------------

    def process_and_move_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file, save its JSON, and move it to processed folder.

        This is the main method for incremental processing. It:
        1. Processes the file through the pipeline
        2. Saves the chunks to a JSON file
        3. Moves the original file to the processed folder

        Args:
            file_path (str): Path to the document file.

        Returns:
            Dict[str, Any]: Dictionary containing processing results:
                - file_name: Name of the processed file
                - status: 'success' or 'failed'
                - chunks: Number of chunks created
                - json_file: Path to the saved JSON file
                - moved_to: New location of the original file

        Example:
            >>> result = pipeline.process_and_move_file("/data/raw/report.pdf")
            >>> print(f"Created {result['chunks']} chunks")
        """
        file_path = Path(file_path)
        file_name = file_path.name

        logger.info(f"Processing and moving file: {file_name}")

        # ----- PROCESS THE FILE -----
        processed_docs = self.process_file(str(file_path))

        # ----- CHECK FOR EMPTY RESULTS -----
        if not processed_docs:
            logger.warning(f"No documents generated for: {file_name}")
            return {
                'file_name': file_name,
                'status': 'failed',
                'chunks': 0,
                'json_file': None,
                'moved_to': None
            }

        # ----- SAVE JSON FOR THIS FILE -----
        json_file = self.save_file_json(processed_docs, file_name)

        # ----- MOVE ORIGINAL FILE TO PROCESSED FOLDER -----
        moved_to = self.move_file_to_processed(str(file_path))

        return {
            'file_name': file_name,
            'status': 'success',
            'chunks': len(processed_docs),
            'json_file': json_file,
            'moved_to': moved_to
        }

    # -------------------------------------------------------------------------
    # DIRECTORY PROCESSING
    # -------------------------------------------------------------------------

    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all documents in a directory through the complete pipeline.

        Loads all supported files from the directory and processes them
        as a single batch.

        Args:
            directory_path (str): Path to directory containing documents.

        Returns:
            List[Document]: List of processed Document objects.

        Example:
            >>> docs = pipeline.process_directory("/data/documents/")
            >>> print(f"Processed {len(docs)} total chunks")
        """
        logger.info(f"Processing directory: {directory_path}")

        # Load all documents from directory
        documents = self.loader.load_documents_from_directory(directory_path)

        # Process through pipeline
        processed_docs = self.process_documents(documents)

        return processed_docs

    def process_raw_documents(self) -> List[Document]:
        """
        Process all raw documents from the configured raw documents directory.

        Uses the RAW_DOCUMENTS_PATH from application settings.

        Returns:
            List[Document]: List of processed Document objects ready for embedding.

        Note:
            This is a convenience method for batch processing all raw documents.
        """
        logger.info("Processing raw documents from configured directory")

        # Load raw documents
        documents = self.loader.load_raw_documents()

        # Process through pipeline
        processed_docs = self.process_documents(documents)

        return processed_docs

    # -------------------------------------------------------------------------
    # BATCH JSON SAVING
    # -------------------------------------------------------------------------

    def save_processed_documents(
        self,
        documents: List[Document],
        output_file: str = None
    ) -> str:
        """
        Save processed documents to a JSON file.

        Saves all documents to a single JSON file for batch processing.

        Args:
            documents (List[Document]): List of processed Document objects.
            output_file (str): Output file path (optional, defaults to
                              processed_documents.json in processed directory).

        Returns:
            str: Path to the saved file.

        Example:
            >>> path = pipeline.save_processed_documents(all_docs)
            >>> print(f"Saved to: {path}")
        """
        # ----- DETERMINE OUTPUT PATH -----
        if output_file is None:
            output_dir = Path(settings.processed_documents_path_absolute)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "processed_documents.json"
        else:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # ----- CONVERT DOCUMENTS TO SERIALIZABLE FORMAT -----
        docs_data = []
        for doc in documents:
            docs_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })

        # ----- SAVE TO JSON -----
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(documents)} processed documents to: {output_file}")
        return str(output_file)

    # -------------------------------------------------------------------------
    # PIPELINE STATISTICS
    # -------------------------------------------------------------------------

    def get_pipeline_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get a summary of the processing pipeline results.

        Generates statistics about the processed documents including
        counts, content types, and source files.

        Args:
            documents (List[Document]): List of processed documents.

        Returns:
            Dict[str, Any]: Dictionary containing pipeline statistics:
                - total_documents: Total number of chunks
                - total_characters: Sum of all characters
                - total_words: Sum of all words
                - avg_chunk_size: Average chunk size in characters
                - content_types: Count by content type
                - chunk_types: Count by chunk type
                - source_files: List of source file names
                - num_source_files: Number of unique source files
        """
        # ----- BASIC STATISTICS -----
        summary = {
            'total_documents': len(documents),
            'total_characters': sum(len(doc.page_content) for doc in documents),
            'total_words': sum(len(doc.page_content.split()) for doc in documents),
            'avg_chunk_size': round(
                sum(len(doc.page_content) for doc in documents) / len(documents)
            ) if documents else 0,
        }

        # ----- COUNT BY CONTENT TYPE -----
        content_types = {}
        for doc in documents:
            content_type = doc.metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        summary['content_types'] = content_types

        # ----- COUNT BY CHUNK TYPE -----
        chunk_types = {}
        for doc in documents:
            chunk_type = doc.metadata.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        summary['chunk_types'] = chunk_types

        # ----- SOURCE FILES -----
        source_files = set()
        for doc in documents:
            file_name = doc.metadata.get('file_name', 'unknown')
            source_files.add(file_name)
        summary['source_files'] = list(source_files)
        summary['num_source_files'] = len(source_files)

        return summary

    # -------------------------------------------------------------------------
    # FILE DISCOVERY
    # -------------------------------------------------------------------------

    def get_raw_files(self) -> List[Path]:
        """
        Get list of all supported files in the raw documents directory.

        Recursively scans all subdirectories for supported file types.

        Returns:
            List[Path]: List of Path objects for each supported file.

        Example:
            >>> files = pipeline.get_raw_files()
            >>> print(f"Found {len(files)} files to process")
        """
        raw_path = Path(settings.raw_documents_path_absolute)

        # Check if directory exists
        if not raw_path.exists():
            return []

        # ----- FIND ALL SUPPORTED FILES -----
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            # Use recursive glob pattern
            files.extend(raw_path.glob(f"**/*{ext}"))

        return files

    # -------------------------------------------------------------------------
    # COMPLETE PIPELINE EXECUTION
    # -------------------------------------------------------------------------

    def run_complete_pipeline(
        self,
        save_output: bool = True,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Run the complete document processing pipeline from start to finish.

        Processes each file individually:
        1. Finds all files in raw directory
        2. Processes each file through the pipeline
        3. Saves separate JSON for each file
        4. Moves processed files to processed directory

        This is the main entry point for processing all raw documents.

        Args:
            save_output (bool): Whether to save processed documents to file.
            output_file (str): Output file path (optional, not used in
                              incremental processing).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - processed_files: List of processing results for each file
                - summary: Pipeline statistics (total_files, successful,
                          failed, total_chunks, processed_files list)

        Example:
            >>> result = pipeline.run_complete_pipeline()
            >>> print(f"Processed {result['summary']['successful']} files")
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting complete document processing pipeline")
            logger.info("=" * 80)

            # ----- GET ALL RAW FILES -----
            raw_files = self.get_raw_files()

            # Handle case when no files found
            if not raw_files:
                logger.warning("No files found in raw documents directory")
                return {
                    'processed_files': [],
                    'summary': {
                        'total_files': 0,
                        'successful': 0,
                        'failed': 0,
                        'total_chunks': 0
                    }
                }

            logger.info(f"Found {len(raw_files)} files to process")

            # ----- PROCESS EACH FILE INDIVIDUALLY -----
            processed_files = []
            all_chunks = []
            successful = 0
            failed = 0

            for file_path in raw_files:
                try:
                    logger.info(f"\n{'='*40}")
                    logger.info(f"Processing: {file_path.name}")
                    logger.info(f"{'='*40}")

                    # Process single file
                    result = self.process_and_move_file(str(file_path))
                    processed_files.append(result)

                    # Track success/failure
                    if result['status'] == 'success':
                        successful += 1
                        all_chunks.extend([result['chunks']])
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {str(e)}")
                    processed_files.append({
                        'file_name': file_path.name,
                        'status': 'failed',
                        'error': str(e),
                        'chunks': 0,
                        'json_file': None,
                        'moved_to': None
                    })
                    failed += 1

            # ----- GENERATE SUMMARY -----
            total_chunks = sum(f['chunks'] for f in processed_files)
            summary = {
                'total_files': len(raw_files),
                'successful': successful,
                'failed': failed,
                'total_chunks': total_chunks,
                'processed_files': [f['file_name'] for f in processed_files if f['status'] == 'success']
            }

            # ----- LOG SUMMARY -----
            logger.info("\n" + "=" * 80)
            logger.info("Pipeline Summary:")
            logger.info(f"  Total files processed: {summary['total_files']}")
            logger.info(f"  Successful: {summary['successful']}")
            logger.info(f"  Failed: {summary['failed']}")
            logger.info(f"  Total chunks created: {summary['total_chunks']}")
            logger.info("=" * 80)

            result = {
                'processed_files': processed_files,
                'summary': summary,
            }

            logger.info("Pipeline completed successfully!")
            logger.info("Documents are ready for embedding generation.")
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise DocumentProcessingError(
                f"Pipeline execution failed: {str(e)}",
                details={'error': str(e)}
            )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for running the pipeline as a standalone script.

    Sets up logging and executes the complete processing pipeline.
    Prints a summary of results to stdout.

    Returns:
        int: Exit code (0 for success, 1 for failure).

    Usage:
        python -m app.services.document_processor.pipeline
    """
    # ----- SET UP LOGGING -----
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        # ----- CREATE PIPELINE -----
        pipeline = DocumentProcessingPipeline(
            use_semantic_chunking=True,
        )

        # ----- RUN COMPLETE PIPELINE -----
        result = pipeline.run_complete_pipeline(save_output=True)

        summary = result['summary']

        # ----- PRINT SUCCESS SUMMARY -----
        print("\n" + "=" * 80)
        print("SUCCESS! Pipeline completed.")
        print("=" * 80)
        print(f"\nTotal files processed: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total chunks created: {summary['total_chunks']}")

        if summary['processed_files']:
            print(f"\nProcessed files:")
            for f in summary['processed_files']:
                print(f"  - {f}")

        print(f"\nJSON results saved to: {settings.processed_documents_path_absolute}/results/")
        print(f"Original files moved to: {settings.processed_documents_path_absolute}/")
        print("\nDocuments are ready for embedding generation.")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

        # ----- PRINT ERROR SUMMARY -----
        print("\n" + "=" * 80)
        print("ERROR! Pipeline failed.")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        return 1


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())
