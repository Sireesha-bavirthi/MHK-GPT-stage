"""
__Project__: Company Chatbot
__Description__: Metadata Extractor Module that extracts and enriches document metadata including file info, text statistics, content type detection, keyword extraction, and chunk positioning for improved retrieval.
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
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from collections import Counter

from langchain_core.documents import Document


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Stop words for keyword extraction (common words to filter out)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
    'which', 'who', 'when', 'where', 'why', 'how'
}


# =============================================================================
# METADATA EXTRACTOR CLASS
# =============================================================================

class MetadataExtractor:
    """
    Metadata Extractor Class.

    Extracts and enriches metadata from documents to improve search and retrieval.
    Provides comprehensive document analysis including statistics, content type
    detection, and keyword extraction.

    Metadata includes:
        - File information (size, type, name)
        - Document statistics (length, word count, etc.)
        - Processing information (timestamps, chunk info)
        - Content type detection (FAQ, technical, list, general)
        - Keywords extracted via frequency analysis

    Example:
        >>> extractor = MetadataExtractor()
        >>> enriched_doc = extractor.enrich_document_metadata(document)
        >>> print(enriched_doc.metadata['keywords'])
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(self):
        """
        Initialize the MetadataExtractor.

        Currently no configuration needed, but provides extensibility
        for future metadata extraction options.
        """
        pass

    # -------------------------------------------------------------------------
    # FILE METADATA EXTRACTION
    # -------------------------------------------------------------------------

    def extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file path.

        Extracts file system information including:
        - File name, stem, and extension
        - File size (bytes, KB, MB)
        - Creation and modification timestamps

        Args:
            file_path (str): Path to the file.

        Returns:
            Dict[str, Any]: Dictionary containing file metadata.

        Example:
            >>> metadata = extractor.extract_file_metadata("/data/report.pdf")
            >>> print(metadata['file_size_mb'])  # 2.5
        """
        path = Path(file_path)
        metadata = {}

        # Only extract if file exists
        if path.exists():
            stat = path.stat()

            # ----- FILE IDENTIFICATION -----
            metadata.update({
                'file_path': str(path),
                'file_name': path.name,
                'file_stem': path.stem,  # Filename without extension
                'file_extension': path.suffix,
            })

            # ----- FILE SIZE (multiple units) -----
            metadata.update({
                'file_size_bytes': stat.st_size,
                'file_size_kb': round(stat.st_size / 1024, 2),
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
            })

            # ----- TIMESTAMPS -----
            metadata.update({
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return metadata

    # -------------------------------------------------------------------------
    # TEXT STATISTICS EXTRACTION
    # -------------------------------------------------------------------------

    def extract_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Extract statistics from text content.

        Calculates various metrics about the text:
        - Character count
        - Word count
        - Line count
        - Paragraph count
        - Sentence count (approximate)
        - Average word length

        Args:
            text (str): Text content to analyze.

        Returns:
            Dict[str, Any]: Dictionary containing text statistics.

        Example:
            >>> stats = extractor.extract_text_statistics("Hello world. How are you?")
            >>> print(stats['word_count'])  # 5
        """
        # ----- BASIC COUNTS -----
        statistics = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }

        # ----- SENTENCE COUNT (approximate using punctuation) -----
        sentences = re.split(r'[.!?]+', text)
        statistics['sentence_count'] = len([s for s in sentences if s.strip()])

        # ----- AVERAGE WORD LENGTH -----
        words = text.split()
        if words:
            statistics['avg_word_length'] = round(sum(len(word) for word in words) / len(words), 2)
        else:
            statistics['avg_word_length'] = 0

        return statistics

    # -------------------------------------------------------------------------
    # CONTENT TYPE DETECTION
    # -------------------------------------------------------------------------

    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of content in the document.

        Analyzes text patterns to classify content as:
        - 'faq': Question and answer format
        - 'technical_documentation': API/code documentation
        - 'list_document': Bullet point or numbered lists
        - 'general': Default for other content

        Args:
            text (str): Document text to analyze.

        Returns:
            str: Content type string ('faq', 'technical_documentation',
                 'list_document', or 'general').

        Example:
            >>> content_type = extractor.detect_content_type("Q: What is AI?\\nA: AI is...")
            >>> print(content_type)  # 'faq'
        """
        # ----- CHECK FOR FAQ PATTERN -----
        faq_markers = [
            r'^\s*Q[:\-\.]',
            r'^\s*A[:\-\.]',
            r'^\s*Question[:\-\.]',
            r'^\s*Answer[:\-\.]',
        ]
        if any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) for pattern in faq_markers):
            return 'faq'

        # ----- CHECK FOR TECHNICAL DOCUMENTATION -----
        tech_markers = [
            r'\bAPI\b',
            r'\bfunction\b',
            r'\bclass\b',
            r'\bmethod\b',
            r'\bparameter\b',
            r'\breturn\b',
            r'```',  # Code blocks
        ]
        tech_count = sum(1 for pattern in tech_markers if re.search(pattern, text, re.IGNORECASE))
        if tech_count >= 3:
            return 'technical_documentation'

        # ----- CHECK FOR LIST/BULLET CONTENT -----
        list_markers = [r'^\s*[-•●○]\s+', r'^\s*\d+\.\s+']
        list_count = sum(len(re.findall(pattern, text, re.MULTILINE)) for pattern in list_markers)
        if list_count > 5:
            return 'list_document'

        # ----- DEFAULT -----
        return 'general'

    # -------------------------------------------------------------------------
    # KEYWORD EXTRACTION
    # -------------------------------------------------------------------------

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text using simple frequency analysis.

        Process:
        1. Tokenize text into words (3+ characters)
        2. Convert to lowercase
        3. Filter out common stop words
        4. Count word frequencies
        5. Return top N most frequent words

        Args:
            text (str): Document text to analyze.
            top_n (int): Number of top keywords to extract (default: 10).

        Returns:
            List[str]: List of extracted keywords sorted by frequency.

        Example:
            >>> keywords = extractor.extract_keywords("AI and machine learning...")
            >>> print(keywords)  # ['machine', 'learning', 'data', ...]
        """
        # ----- TOKENIZE AND CLEAN -----
        # Extract words with 3+ characters, convert to lowercase
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # ----- FILTER OUT STOP WORDS -----
        filtered_words = [word for word in words if word not in STOP_WORDS]

        # ----- COUNT FREQUENCIES -----
        word_counts = Counter(filtered_words)

        # ----- GET TOP N KEYWORDS -----
        keywords = [word for word, _ in word_counts.most_common(top_n)]

        return keywords

    # -------------------------------------------------------------------------
    # SINGLE DOCUMENT ENRICHMENT
    # -------------------------------------------------------------------------

    def enrich_document_metadata(self, document: Document) -> Document:
        """
        Enrich a document with additional metadata.

        Adds the following metadata to the document:
        - Text statistics (char_count, word_count, etc.)
        - Content type detection
        - Extracted keywords
        - Processing timestamp
        - Content hash for deduplication

        Args:
            document (Document): LangChain Document object.

        Returns:
            Document: New Document with enriched metadata.

        Note:
            Original document is not modified; a new document is created.
        """
        text = document.page_content
        metadata = document.metadata.copy()

        # ----- STEP 1: Extract text statistics -----
        stats = self.extract_text_statistics(text)
        metadata.update(stats)

        # ----- STEP 2: Detect content type -----
        content_type = self.detect_content_type(text)
        metadata['content_type'] = content_type

        # ----- STEP 3: Extract keywords -----
        keywords = self.extract_keywords(text, top_n=10)
        metadata['keywords'] = keywords

        # ----- STEP 4: Add processing timestamp -----
        metadata['processed_at'] = datetime.now().isoformat()

        # ----- STEP 5: Calculate content hash for deduplication -----
        content_hash = hashlib.md5(text.encode()).hexdigest()
        metadata['content_hash'] = content_hash

        # ----- CREATE ENRICHED DOCUMENT -----
        enriched_doc = Document(
            page_content=text,
            metadata=metadata
        )

        return enriched_doc

    # -------------------------------------------------------------------------
    # BATCH DOCUMENT ENRICHMENT
    # -------------------------------------------------------------------------

    def enrich_documents_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Enrich multiple documents with metadata.

        Applies metadata enrichment to each document in the list.
        If enrichment fails for a document, the original is kept.

        Args:
            documents (List[Document]): List of Document objects.

        Returns:
            List[Document]: List of documents with enriched metadata.

        Example:
            >>> enriched = extractor.enrich_documents_metadata(raw_docs)
            >>> print(f"Enriched {len(enriched)} documents")
        """
        enriched_docs = []

        for i, doc in enumerate(documents):
            try:
                enriched_doc = self.enrich_document_metadata(doc)
                enriched_docs.append(enriched_doc)
            except Exception as e:
                # Log warning but keep original document if enrichment fails
                logger.warning(f"Error enriching metadata for document {i}: {str(e)}")
                enriched_docs.append(doc)

        logger.info(f"Enriched metadata for {len(enriched_docs)} documents")
        return enriched_docs

    # -------------------------------------------------------------------------
    # CHUNK METADATA EXTRACTION
    # -------------------------------------------------------------------------

    def extract_chunk_metadata(self, chunk: Document, chunk_index: int, total_chunks: int) -> Document:
        """
        Add chunk-specific metadata.

        Adds positional information to a chunk:
        - chunk_index: Position in the sequence
        - total_chunks: Total number of chunks
        - is_first_chunk: Boolean flag
        - is_last_chunk: Boolean flag

        Args:
            chunk (Document): Document chunk.
            chunk_index (int): Index of this chunk (0-based).
            total_chunks (int): Total number of chunks in the document.

        Returns:
            Document: Chunk with added positional metadata.

        Example:
            >>> chunk = extractor.extract_chunk_metadata(doc_chunk, 0, 10)
            >>> print(chunk.metadata['is_first_chunk'])  # True
        """
        metadata = chunk.metadata.copy()

        # ----- ADD CHUNK POSITION INFO -----
        metadata.update({
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'is_first_chunk': chunk_index == 0,
            'is_last_chunk': chunk_index == total_chunks - 1,
        })

        # ----- CREATE NEW DOCUMENT WITH UPDATED METADATA -----
        chunk_doc = Document(
            page_content=chunk.page_content,
            metadata=metadata
        )

        return chunk_doc
