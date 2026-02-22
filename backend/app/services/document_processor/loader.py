"""
__Project__: Company Chatbot
__Description__: Document Loader Module that handles loading documents from various file formats (PDF, DOCX, TXT, MD) using LangChain loaders. Provides unified interface for single file and directory loading with metadata enrichment.
__Created Date__: 04-02-2026
__Updated Date__: 06-02-2026
__Author__: Nagamani Bhukya
__Employee Id__: 800339
"""

# =============================================================================
# IMPORTS
# =============================================================================

from typing import List
from pathlib import Path
import logging

from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError, InvalidFileTypeError


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Initialize logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# DOCUMENT LOADER CLASS
# =============================================================================

class DocumentLoader:
    """
    Document Loader Class.

    Loads documents from various file formats using LangChain document loaders.
    Supports recursive directory scanning and metadata enrichment.

    Attributes:
        supported_extensions (dict): Mapping of file extensions to their loaders.

    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_document("report.pdf")
        >>> print(f"Loaded {len(docs)} pages")
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self):
        """
        Initialize the DocumentLoader with supported file type mappings.

        Sets up the mapping between file extensions and their corresponding
        LangChain loader classes.
        """
        # Define supported file extensions and their corresponding loaders
        self.supported_extensions = {
            '.docx': Docx2txtLoader,    # Microsoft Word documents
            '.txt': TextLoader,          # Plain text files
            '.md': UnstructuredMarkdownLoader,  # Markdown files
            '.pdf': PyPDFLoader,         # PDF documents
        }

    # -------------------------------------------------------------------------
    # SINGLE FILE LOADING
    # -------------------------------------------------------------------------

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document using the appropriate LangChain loader.

        This method:
        1. Validates the file exists
        2. Checks if the file type is supported
        3. Loads the document using the appropriate loader
        4. Enriches the document with metadata

        Args:
            file_path (str): Absolute or relative path to the document file.

        Returns:
            List[Document]: List of LangChain Document objects.
                           PDFs return one Document per page.
                           Other formats typically return one Document.

        Raises:
            InvalidFileTypeError: If the file extension is not supported.
            DocumentProcessingError: If the file doesn't exist or loading fails.

        Example:
            >>> docs = loader.load_document("/data/report.pdf")
            >>> for doc in docs:
            ...     print(doc.page_content[:100])
        """
        # Convert to Path object for easier manipulation
        path = Path(file_path)

        # ----- VALIDATION: Check if file exists -----
        if not path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                details={"file_path": file_path}
            )

        # ----- VALIDATION: Check file extension -----
        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise InvalidFileTypeError(
                f"Unsupported file type: {extension}",
                details={"file_path": file_path, "extension": extension}
            )

        # Get file size for metadata (no size limit enforced)
        file_size = path.stat().st_size

        try:
            # ----- LOADING: Get appropriate loader and load document -----
            loader_class = self.supported_extensions[extension]
            loader = loader_class(str(path))

            # Load the document content
            documents = loader.load()

            # ----- METADATA: Enrich documents with source information -----
            for doc in documents:
                doc.metadata.update({
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": extension,
                    "file_size": file_size,
                })

            logger.info(f"Successfully loaded document: {path.name} ({len(documents)} pages/sections)")
            return documents

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to load document: {str(e)}",
                details={"file_path": file_path, "error": str(e)}
            )

    # -------------------------------------------------------------------------
    # DIRECTORY LOADING
    # -------------------------------------------------------------------------

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory and its subdirectories.

        Recursively scans the directory for all supported file types and
        loads them using the appropriate loaders.

        Args:
            directory_path (str): Path to the directory containing documents.

        Returns:
            List[Document]: Combined list of all loaded Document objects.

        Raises:
            DocumentProcessingError: If directory doesn't exist or no documents loaded.

        Example:
            >>> docs = loader.load_documents_from_directory("/data/documents/")
            >>> print(f"Loaded {len(docs)} total documents")
        """
        dir_path = Path(directory_path)

        # ----- VALIDATION: Check directory exists -----
        if not dir_path.exists():
            raise DocumentProcessingError(
                f"Directory not found: {directory_path}",
                details={"directory_path": directory_path}
            )

        # ----- VALIDATION: Check path is a directory -----
        if not dir_path.is_dir():
            raise DocumentProcessingError(
                f"Path is not a directory: {directory_path}",
                details={"directory_path": directory_path}
            )

        # Initialize tracking lists
        all_documents = []
        loaded_files = []
        failed_files = []

        # ----- SCANNING: Find all supported files recursively -----
        # Create glob patterns for each supported extension (e.g., "**/*.pdf")
        supported_patterns = [f"**/*{ext}" for ext in self.supported_extensions.keys()]

        # Process each file type pattern
        for pattern in supported_patterns:
            for file_path in dir_path.glob(pattern):
                try:
                    # Load the document
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                    loaded_files.append(file_path.name)
                except Exception as e:
                    # Log warning but continue processing other files
                    logger.warning(f"Failed to load {file_path.name}: {str(e)}")
                    failed_files.append((file_path.name, str(e)))

        # ----- LOGGING: Report loading results -----
        logger.info(
            f"Loaded {len(loaded_files)} files successfully. "
            f"Failed: {len(failed_files)} files. "
            f"Total documents: {len(all_documents)}"
        )

        if failed_files:
            logger.warning(f"Failed files: {[f[0] for f in failed_files]}")

        # ----- VALIDATION: Ensure at least one document loaded -----
        if not all_documents:
            raise DocumentProcessingError(
                "No documents were successfully loaded",
                details={
                    "directory_path": directory_path,
                    "failed_files": failed_files
                }
            )

        return all_documents

    # -------------------------------------------------------------------------
    # RAW DOCUMENTS LOADING (CONVENIENCE METHOD)
    # -------------------------------------------------------------------------

    def load_raw_documents(self) -> List[Document]:
        """
        Load all raw documents from the configured raw documents directory.

        This is a convenience method that loads documents from the path
        specified in the application settings (RAW_DOCUMENTS_PATH).
        Returns:
            List[Document]: All loaded Document objects from the raw directory.
        Note:
            The raw documents path is configured in app.core.config.settings
        """
        raw_docs_path = settings.raw_documents_path_absolute
        logger.info(f"Loading raw documents from: {raw_docs_path}")
        return self.load_documents_from_directory(raw_docs_path)
