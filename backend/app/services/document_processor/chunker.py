"""
__Project__: Company Chatbot
__Description__: Document Chunker Module that implements intelligent text chunking using semantic similarity and FAQ detection. Groups related sentences using OpenAI embeddings and keeps Q&A pairs together for better RAG retrieval.
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
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError


# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Initialize logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# CHUNKER CLASS
# =============================================================================

class Chunker:
    """
    Document Chunker Class.

    Implements semantic similarity-based text chunking using embeddings.
    Handles both regular content (semantic chunking) and FAQ content (Q&A pairs together).

    How it works:
        1. Splits text into sentences
        2. Generates embeddings for each sentence
        3. Calculates semantic similarity between consecutive sentences
        4. Groups sentences with high similarity into chunks
        5. Handles FAQ documents specially (keeps Q&A together)

    Attributes:
        similarity_threshold (float): Minimum similarity to group sentences (0-1).
        min_chunk_size (int): Minimum characters per chunk.
        max_chunk_size (int): Maximum characters per chunk.
        embeddings: OpenAI embeddings instance.

    Example:
        >>> chunker = Chunker(similarity_threshold=0.75)
        >>> chunks = chunker.chunk_document(document)
        >>> print(f"Created {len(chunks)} chunks")
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(
        self,
        similarity_threshold: float = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
    ):
        """
        Initialize the Chunker with configurable parameters.

        Args:
            similarity_threshold (float, optional): Threshold for semantic similarity (0-1).
                Defaults to settings.SEMANTIC_SIMILARITY_THRESHOLD.
            min_chunk_size (int, optional): Minimum chunk size in characters.
                Defaults to settings.SEMANTIC_MIN_CHUNK_SIZE.
            max_chunk_size (int, optional): Maximum chunk size in characters.
                Defaults to settings.SEMANTIC_MAX_CHUNK_SIZE.

        Raises:
            DocumentProcessingError: If embeddings initialization fails.
        """
        # Set chunking parameters (use settings defaults if not provided)
        self.similarity_threshold = similarity_threshold or settings.SEMANTIC_SIMILARITY_THRESHOLD
        self.min_chunk_size = min_chunk_size or settings.SEMANTIC_MIN_CHUNK_SIZE
        self.max_chunk_size = max_chunk_size or settings.SEMANTIC_MAX_CHUNK_SIZE

        # ----- Initialize OpenAI embeddings -----
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_EMBEDDING_MODEL
            )
            logger.info(f"Initialized OpenAI embeddings with model: {settings.OPENAI_EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to initialize embeddings: {str(e)}",
                details={"error": str(e)}
            )

    # -------------------------------------------------------------------------
    # TEXT SPLITTING
    # -------------------------------------------------------------------------

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.

        Uses regex to split on sentence boundaries while handling
        common abbreviations and edge cases.

        Args:
            text (str): Text to split into sentences.

        Returns:
            List[str]: List of sentences (empty/short sentences filtered out).

        Note:
            Sentences shorter than 10 characters are filtered out.
        """
        # Split on sentence boundaries (period/exclamation/question followed by space and capital)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Filter out empty sentences and very short ones (< 10 chars)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    # -------------------------------------------------------------------------
    # FAQ DETECTION AND EXTRACTION
    # -------------------------------------------------------------------------

    def detect_faq_pattern(self, text: str) -> bool:
        """
        Detect if text contains FAQ pattern (questions and answers).

        Looks for common Q&A markers like "Q:", "Question:", or
        question words followed by question marks.

        Args:
            text (str): Text to analyze.

        Returns:
            bool: True if FAQ pattern detected, False otherwise.
        """
        # Define question markers to look for
        question_markers = [
            r'\?',                    # Contains question marks
            r'^Q[:\-\.]',             # Starts with Q: or Q- or Q.
            r'^Question[:\-\.]',      # Starts with Question:
            # Starts with question words
            r'^What |^How |^Why |^When |^Where |^Who |^Can |^Is |^Are |^Do |^Does ',
        ]

        # Define answer markers to look for
        answer_markers = [
            r'^A[:\-\.]',             # Starts with A: or A- or A.
            r'^Answer[:\-\.]',        # Starts with Answer:
        ]

        # Check if any question or answer markers are present
        has_questions = any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) for pattern in question_markers)
        has_answers = any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) for pattern in answer_markers)

        return has_questions or has_answers

    def extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract question-answer pairs from FAQ text.

        Handles multiple Q&A formats:
            1. Q: Question? A: Answer
            2. Question: ... Answer: ...
            3. What is X? X is... (question ends with ?, answer follows)

        Args:
            text (str): Text containing Q&A pairs.

        Returns:
            List[Tuple[str, str]]: List of (question, answer) tuples.
        """
        qa_pairs = []

        # ----- Try Q:/Question: and A:/Answer: format first -----
        qa_pattern = r'(?:^|\n)(?:Q[\:\-\.\s]|Question[\:\-\.\s])(.*?)(?=(?:^|\n)(?:A[\:\-\.\s]|Answer[\:\-\.\s])|$)'
        answer_pattern = r'(?:^|\n)(?:A[\:\-\.\s]|Answer[\:\-\.\s])(.*?)(?=(?:^|\n)(?:Q[\:\-\.\s]|Question[\:\-\.\s])|$)'

        questions = re.findall(qa_pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        answers = re.findall(answer_pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)

        # Pair questions with answers
        for i, q in enumerate(questions):
            a = answers[i] if i < len(answers) else ""
            if q.strip() and a.strip():
                qa_pairs.append((q.strip(), a.strip()))

        # ----- If no Q:/A: format found, try "Question? Answer" format -----
        if not qa_pairs:
            qa_pairs = self._extract_question_answer_format(text)

        return qa_pairs

    def _extract_question_answer_format(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract Q&A pairs where questions end with ? and answers follow.

        Handles format like: "What is X? X is a technology that..."

        Args:
            text (str): Text containing Q&A pairs.

        Returns:
            List[Tuple[str, str]]: List of (question, answer) tuples.
        """
        qa_pairs = []

        # ----- Remove FAQ section headers (not the content) -----
        # Match: "SECTION 4 - FAQ" or "SECTION 4 - (FAQ - some text)"
        text = re.sub(r'SECTION\s*\d*\s*[-–—:\s]*\s*\(?FAQ[^)\n]*\)?\s*[-–—:\s]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r"Let's dive into[^\n]*\n?", '', text, flags=re.IGNORECASE)

        # ----- Define question starter words -----
        question_starters = r'(?:What|How|Why|When|Where|Who|Can|Is|Are|Do|Does|Which|Would|Should|Could|Will)'

        # Pattern: question word at start of line, followed by text until ?
        question_pattern = rf'(?:^|\n+|\.\s+)({question_starters}[^?]+\?)'

        # ----- Find all questions with their positions -----
        questions_with_pos = []
        for match in re.finditer(question_pattern, text, re.IGNORECASE | re.MULTILINE):
            questions_with_pos.append({
                'question': match.group(1).strip(),
                'start': match.start(1),
                'end': match.end(1)
            })

        # ----- Extract Q&A pairs by getting text between questions -----
        for i, q_info in enumerate(questions_with_pos):
            question = q_info['question']

            # Get answer: text from end of question to start of next question
            answer_start = q_info['end']
            if i + 1 < len(questions_with_pos):
                answer_end = questions_with_pos[i + 1]['start']
            else:
                answer_end = len(text)

            answer = text[answer_start:answer_end].strip()

            # Clean up the answer (remove leading punctuation)
            answer = re.sub(r'^\s*[\.\,\;\:]\s*', '', answer)
            answer = answer.strip()

            if question and answer:
                qa_pairs.append((question, answer))

        return qa_pairs

    # -------------------------------------------------------------------------
    # EMBEDDING GENERATION
    # -------------------------------------------------------------------------

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using OpenAI.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            np.ndarray: NumPy array of embeddings (shape: [n_texts, embedding_dim]).

        Raises:
            DocumentProcessingError: If embedding generation fails.
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to generate embeddings: {str(e)}",
                details={"error": str(e)}
            )

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix for embeddings.

        Args:
            embeddings (np.ndarray): Array of embeddings.

        Returns:
            np.ndarray: Similarity matrix where [i][j] is similarity between i and j.
        """
        return cosine_similarity(embeddings)

    # -------------------------------------------------------------------------
    # SEMANTIC CHUNKING
    # -------------------------------------------------------------------------

    def create_semantic_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create chunks based on semantic similarity between sentences.

        Groups consecutive sentences with high similarity together,
        respecting min/max chunk size constraints.

        Args:
            sentences (List[str]): List of sentences to chunk.

        Returns:
            List[str]: List of chunks (each chunk is a string of grouped sentences).

        Note:
            Splitting occurs when:
            - Similarity drops below threshold
            - Max chunk size would be exceeded
            - Current chunk is above min size and similarity is moderate (<0.85)
        """
        # Handle edge cases
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # ----- Generate embeddings for all sentences -----
        logger.info(f"Generating embeddings for {len(sentences)} sentences...")
        embeddings = self.generate_embeddings(sentences)

        # ----- Calculate similarity matrix -----
        similarity_matrix = self.calculate_similarity_matrix(embeddings)

        # ----- Group sentences based on similarity -----
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_size = len(sentences[0])

        for i in range(1, len(sentences)):
            prev_idx = i - 1
            similarity = similarity_matrix[prev_idx][i]

            # Calculate potential new chunk size
            sentence_size = len(sentences[i])
            potential_size = current_chunk_size + sentence_size

            # Determine if we should split to a new chunk
            should_split = (
                similarity < self.similarity_threshold           # Low similarity
                or potential_size > self.max_chunk_size          # Would exceed max size
                or (current_chunk_size >= self.min_chunk_size and similarity < 0.85)  # Moderate similarity, sufficient size
            )

            if should_split and current_chunk_size >= self.min_chunk_size:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_chunk_size = sentence_size
            else:
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_chunk_size += sentence_size

        # ----- Add the last chunk -----
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks

    # -------------------------------------------------------------------------
    # DOCUMENT CHUNKING
    # -------------------------------------------------------------------------

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document using the appropriate strategy.

        For FAQ sections: Keeps each question-answer pair together in one chunk.
        For other content: Uses semantic similarity chunking.

        Args:
            document (Document): LangChain Document object to chunk.

        Returns:
            List[Document]: List of Document objects (one per chunk).
        """
        text = document.page_content

        # ----- Check for FAQ sections -----
        faq_sections = self._extract_faq_sections(text)

        if faq_sections:
            logger.info(f"FAQ sections detected in document: {document.metadata.get('file_name', 'unknown')}")
            chunks = []
            chunk_index = 0

            for section_text, is_faq in faq_sections:
                if is_faq:
                    # ----- FAQ Section: Extract Q&A pairs -----
                    qa_pairs = self.extract_qa_pairs(section_text)
                    for question, answer in qa_pairs:
                        # Keep question and answer together in one chunk
                        chunk_text = f"Q: {question}\n\nA: {answer}"
                        chunk_doc = Document(
                            page_content=chunk_text,
                            metadata={
                                **document.metadata,
                                'chunk_index': chunk_index,
                                'chunk_type': 'faq',
                            }
                        )
                        chunks.append(chunk_doc)
                        chunk_index += 1
                else:
                    # ----- Non-FAQ Section: Semantic chunking -----
                    section_chunks = self._chunk_regular_text(section_text, document.metadata, chunk_index)
                    for chunk in section_chunks:
                        chunk.metadata['chunk_index'] = chunk_index
                        chunks.append(chunk)
                        chunk_index += 1

            # Update total_chunks in all chunks
            for chunk in chunks:
                chunk.metadata['total_chunks'] = len(chunks)

            return chunks if chunks else [document]

        # ----- No FAQ detected: Use regular semantic chunking -----
        return self._chunk_regular_text(text, document.metadata, 0)

    def _extract_faq_sections(self, text: str) -> List[Tuple[str, bool]]:
        """
        Extract FAQ and non-FAQ sections from text.

        Only treats explicitly marked FAQ sections as FAQ.
        All other content uses semantic chunking.

        Args:
            text (str): Full document text.

        Returns:
            List[Tuple[str, bool]]: List of (section_text, is_faq) tuples.
                Empty list means no FAQ markers found.
        """
        # Define explicit FAQ section markers
        # Note: [-–—:] includes regular hyphen, en-dash, em-dash, and colon
        faq_markers = [
            r'(?i)SECTION\s*\d*\s*[-–—:\s]*\s*\(?FAQ[^)\n]*\)?',
            r'(?i)^FAQ\s*$',
            r'(?i)^Frequently Asked Questions',
        ]

        for marker in faq_markers:
            match = re.search(marker, text, re.MULTILINE)
            if match:
                # Split into before FAQ and FAQ section
                before_faq = text[:match.start()].strip()
                faq_section = text[match.start():].strip()

                sections = []
                if before_faq:
                    sections.append((before_faq, False))   # Non-FAQ -> semantic chunking
                if faq_section:
                    sections.append((faq_section, True))   # FAQ -> Q&A chunking
                return sections

        # No FAQ marker found - return empty (semantic chunking for all)
        return []

    def _chunk_regular_text(self, text: str, base_metadata: dict, start_index: int) -> List[Document]:
        """
        Chunk regular (non-FAQ) text using semantic similarity.

        Args:
            text (str): Text to chunk.
            base_metadata (dict): Base metadata to include in chunks.
            start_index (int): Starting chunk index.

        Returns:
            List[Document]: List of Document objects (chunks).
        """
        sentences = self.split_into_sentences(text)

        # Handle case where no sentences could be extracted
        if not sentences:
            return [Document(
                page_content=text,
                metadata={
                    **base_metadata,
                    'chunk_index': start_index,
                    'chunk_type': 'semantic',
                    'total_chunks': 1,
                    'chunk_size': len(text)
                }
            )]

        # ----- Create semantic chunks -----
        chunk_texts = self.create_semantic_chunks(sentences)

        # ----- Create Document objects for each chunk -----
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **base_metadata,
                    'chunk_index': start_index + i,
                    'chunk_type': 'semantic',
                    'total_chunks': len(chunk_texts),
                    'chunk_size': len(chunk_text)
                }
            )
            chunks.append(chunk_doc)

        return chunks

    # -------------------------------------------------------------------------
    # BATCH DOCUMENT CHUNKING
    # -------------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents using semantic similarity.

        Processes each document and returns all chunks combined.
        If chunking fails for a document, the original is kept.

        Args:
            documents (List[Document]): List of Document objects to chunk.

        Returns:
            List[Document]: List of all chunked Document objects.

        Example:
            >>> chunker = Chunker()
            >>> all_chunks = chunker.chunk_documents(documents)
            >>> print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        """
        all_chunks = []

        for doc_idx, document in enumerate(documents):
            try:
                logger.info(f"Chunking document {doc_idx + 1}/{len(documents)}: {document.metadata.get('file_name', 'unknown')}")
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc_idx}: {str(e)}")
                # Keep original document if chunking fails
                all_chunks.append(document)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
