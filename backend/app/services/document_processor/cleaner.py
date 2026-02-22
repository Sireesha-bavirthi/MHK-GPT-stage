"""
__Project__: Company Chatbot
__Description__: Text Cleaner Module that provides text cleaning and normalization functionality for documents. Removes noise, fixes formatting, normalizes unicode characters, and filters empty documents before chunking.
__Created Date__: 04-02-2026
__Updated Date__: 06-02-2026
__Author__: Nagamani Bhukya
__Employee Id__: 800339
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
import logging
from typing import List

from langchain_core.documents import Document


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Minimum character count for a valid document (used in filtering)
MIN_DOCUMENT_LENGTH = 10


# =============================================================================
# TEXT CLEANER CLASS
# =============================================================================

class TextCleaner:
    """
    Text Cleaner Class.

    Cleans and normalizes text from documents before chunking and embedding.
    Handles whitespace, special characters, and formatting issues.

    Example:
        >>> cleaner = TextCleaner()
        >>> cleaned_text = cleaner.clean_text("  Hello   World!  ")
        >>> print(cleaned_text)  # "Hello World!"
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self):
        """
        Initialize the TextCleaner.

        Currently no configuration needed, but provides extensibility
        for future cleaning options.
        """
        pass

    # -------------------------------------------------------------------------
    # CORE TEXT CLEANING
    # -------------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a single text string.

        Performs multiple cleaning operations to prepare text for processing:
        1. Remove URLs
        2. Normalize unicode characters (quotes, dashes, ellipsis)
        3. Remove excessive whitespace
        4. Remove control characters
        5. Fix punctuation spacing
        6. Normalize bullet points
        7. Remove page numbers

        Args:
            text (str): Raw text to clean.

        Returns:
            str: Cleaned and normalized text string.

        Example:
            >>> cleaner = TextCleaner()
            >>> raw = "  Hello   World!   Check http://example.com  "
            >>> cleaner.clean_text(raw)
            'Hello World! Check'
        """
        # Handle empty input
        if not text:
            return ""

        # ----- STEP 1: Remove URLs -----
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # ----- STEP 2: Normalize unicode characters -----
        # Smart single quotes -> regular single quotes
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        # Smart double quotes -> regular double quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        # En dash and em dash -> regular dash
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        # Ellipsis character -> three dots
        text = text.replace('\u2026', '...')

        # ----- STEP 3: Remove excessive whitespace -----
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Reduce multiple newlines to double newline (preserve paragraph breaks)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # ----- STEP 4: Strip whitespace from each line -----
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        # ----- STEP 5: Remove control characters -----
        # Keep only printable characters plus newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])

        # ----- STEP 6: Fix punctuation spacing -----
        # Remove space before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        # Add space after punctuation if missing (before a letter)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        # Remove multiple consecutive punctuation (except ellipsis)
        text = re.sub(r'([.,;:!?])\1+', r'\1', text)

        # ----- STEP 7: Normalize bullet points -----
        # Convert various bullet characters to standard dash
        text = re.sub(r'^\s*[•●○■◆▪▸►‣⁃-]\s*', '- ', text, flags=re.MULTILINE)

        # ----- STEP 8: Remove page numbers and standalone numbers -----
        # Remove "Page X" patterns
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        # Remove standalone numbers (likely page numbers)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # ----- FINAL: Strip leading/trailing whitespace -----
        text = text.strip()

        return text

    # -------------------------------------------------------------------------
    # DOCUMENT CLEANING
    # -------------------------------------------------------------------------

    def clean_document(self, document: Document) -> Document:
        """
        Clean text content of a LangChain Document object.

        Creates a new Document with cleaned text while preserving metadata.

        Args:
            document (Document): LangChain Document object with raw text.

        Returns:
            Document: New Document object with cleaned text and copied metadata.

        Note:
            Original document is not modified; a new document is created.
        """
        # Clean the text content
        cleaned_text = self.clean_text(document.page_content)

        # Create new document with cleaned text (preserve metadata)
        cleaned_doc = Document(
            page_content=cleaned_text,
            metadata=document.metadata.copy()
        )

        return cleaned_doc

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean multiple documents.

        Applies cleaning to each document in the list. If cleaning fails
        for a document, the original is kept with a warning logged.

        Args:
            documents (List[Document]): List of LangChain Document objects.

        Returns:
            List[Document]: List of cleaned Document objects.

        Example:
            >>> cleaned = cleaner.clean_documents(raw_documents)
            >>> print(f"Cleaned {len(cleaned)} documents")
        """
        cleaned_docs = []

        for i, doc in enumerate(documents):
            try:
                cleaned_doc = self.clean_document(doc)
                cleaned_docs.append(cleaned_doc)
            except Exception as e:
                # Log warning but keep original document if cleaning fails
                logger.warning(f"Error cleaning document {i}: {str(e)}")
                cleaned_docs.append(doc)

        logger.info(f"Cleaned {len(cleaned_docs)} documents")
        return cleaned_docs

    # -------------------------------------------------------------------------
    # DOCUMENT FILTERING
    # -------------------------------------------------------------------------

    def remove_empty_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove documents with empty or very short content.

        Filters out documents that have insufficient content after cleaning.
        Uses MIN_DOCUMENT_LENGTH constant to determine minimum valid length.

        Args:
            documents (List[Document]): List of Document objects to filter.

        Returns:
            List[Document]: List of non-empty Document objects.

        Note:
            Documents with less than MIN_DOCUMENT_LENGTH characters are removed.
        """
        # Filter documents with sufficient content
        non_empty_docs = [
            doc for doc in documents
            if doc.page_content and len(doc.page_content.strip()) >= MIN_DOCUMENT_LENGTH
        ]

        # Log if any documents were removed
        removed_count = len(documents) - len(non_empty_docs)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} empty or very short documents")

        return non_empty_docs

    # -------------------------------------------------------------------------
    # COMBINED OPERATIONS
    # -------------------------------------------------------------------------

    def clean_and_filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean documents and remove empty ones in a single operation.

        This is the main entry point for document cleaning. It:
        1. Cleans all documents (normalizes text)
        2. Filters out empty/short documents

        Args:
            documents (List[Document]): List of raw Document objects.

        Returns:
            List[Document]: List of cleaned, non-empty Document objects.

        Example:
            >>> cleaner = TextCleaner()
            >>> processed = cleaner.clean_and_filter_documents(raw_docs)
            >>> print(f"Processed {len(processed)} valid documents")
        """
        # Step 1: Clean all documents
        cleaned_docs = self.clean_documents(documents)

        # Step 2: Remove empty/short documents
        filtered_docs = self.remove_empty_documents(cleaned_docs)

        return filtered_docs
