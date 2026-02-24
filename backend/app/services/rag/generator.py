"""
__Project__: Company Chatbot
__Description__: Response generation service for RAG chatbot. Combines retrieval results with LLM to generate answers.
__Created Date__: 05-02-2026
__Updated Date__: 06-02-2026
__Author__: SWATHI KAVITI
__Employee Id__: 800340

"""

import logging
import time
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass

from app.services.rag.retriever import RetrievalResult
from app.services.llm.openai_client import get_openai_client, OpenAIClient
from app.services.llm.prompt_templates import (
    format_context_from_results,
    format_system_prompt,
    format_user_prompt,
    build_messages,
    truncate_conversation_history,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GenerationResult:
    """Result from response generation."""
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    model: str


# =============================================================================
# Generator Service
# =============================================================================

class GeneratorService:
    """
    Response generation service.
    Generates answers using retrieved context and conversation history.
    """
    
    def __init__(
        self,
        openai_client: Optional[OpenAIClient] = None,
        company_name: Optional[str] = None,
        max_conversation_history: int = 10
    ):
        """
        Initialize generator service.
        
        Args:
            openai_client: OpenAI client instance (default: singleton)
            company_name: Company name for prompts (default from settings)
            max_conversation_history: Max messages to keep in history
        """
        self.openai_client = openai_client or get_openai_client()
        self.company_name = company_name or getattr(settings, 'COMPANY_NAME', 'MHKTech')
        self.max_conversation_history = max_conversation_history
        
        # Statistics
        self.total_generations = 0
        self.total_tokens_used = 0
        self.total_generation_time = 0.0
        
        logger.info(
            f"GeneratorService initialized | company={self.company_name} | "
            f"max_history={self.max_conversation_history}"
        )
    
    def generate(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> GenerationResult:
        """
        Generate response from query and retrieval results.
        
        Args:
            query: User's question
            retrieval_results: Retrieved document chunks
            conversation_history: Previous conversation messages
            stream: Whether to stream response
            
        Returns:
            GenerationResult with response and metadata
        """
        if stream:
            raise ValueError("Use generate_stream() for streaming responses")
        
        start_time = time.time()
        
        # Format context from retrieval results
        context = format_context_from_results(retrieval_results)
        logger.debug(f"Formatted context from {len(retrieval_results)} results")
        
        # Build prompts
        system_prompt = format_system_prompt(self.company_name, context)
        user_prompt = format_user_prompt(query)
        
        # Truncate conversation history if needed
        if conversation_history:
            conversation_history = truncate_conversation_history(
                conversation_history,
                self.max_conversation_history
            )
            logger.debug(f"Using {len(conversation_history)} messages from history")
        
        # Build complete message list
        messages = build_messages(system_prompt, user_prompt, conversation_history)
        
        logger.info(
            f"Generating response | query='{query[:50]}...' | "
            f"context_chunks={len(retrieval_results)} | "
            f"history_messages={len(conversation_history) if conversation_history else 0}"
        )
        
        # Generate response
        try:
            response = self.openai_client.chat_completion(messages=messages)
            
            elapsed = time.time() - start_time
            
            # Extract response and usage
            content = response.choices[0].message.content
            usage = response.usage
            
            # Update statistics
            self.total_generations += 1
            self.total_tokens_used += usage.total_tokens
            self.total_generation_time += elapsed
            
            # Log detailed info
            logger.info(
                f"Response generated | "
                f"prompt_tokens={usage.prompt_tokens} | "
                f"completion_tokens={usage.completion_tokens} | "
                f"total_tokens={usage.total_tokens} | "
                f"time={elapsed:.2f}s"
            )
            
            # Print to terminal for visibility
            print(f"\n{'='*80}")
            print(f"GENERATION COMPLETE")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Context Chunks: {len(retrieval_results)}")
            print(f"Prompt Tokens: {usage.prompt_tokens}")
            print(f"Completion Tokens: {usage.completion_tokens}")
            print(f"Total Tokens: {usage.total_tokens}")
            print(f"Generation Time: {elapsed:.2f}s")
            print(f"{'='*80}\n")
            
            return GenerationResult(
                response=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                generation_time=elapsed,
                model=response.model
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise
    
    def generate_stream(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Iterator[str]:
        """
        Generate streaming response from query and retrieval results.
        
        Args:
            query: User's question
            retrieval_results: Retrieved document chunks
            conversation_history: Previous conversation messages
            
        Yields:
            Response content chunks as they arrive
        """
        start_time = time.time()
        
        # Format context from retrieval results
        context = format_context_from_results(retrieval_results)
        logger.debug(f"Formatted context from {len(retrieval_results)} results")
        
        # Build prompts
        system_prompt = format_system_prompt(self.company_name, context)
        user_prompt = format_user_prompt(query)
        
        # Truncate conversation history if needed
        if conversation_history:
            conversation_history = truncate_conversation_history(
                conversation_history,
                self.max_conversation_history
            )
            logger.debug(f"Using {len(conversation_history)} messages from history")
        
        # Build complete message list
        messages = build_messages(system_prompt, user_prompt, conversation_history)
        
        logger.info(
            f"Generating streaming response | query='{query[:50]}...' | "
            f"context_chunks={len(retrieval_results)} | "
            f"history_messages={len(conversation_history) if conversation_history else 0}"
        )
        
        # Print header to terminal
        print(f"\n{'='*80}")
        print(f"STREAMING GENERATION STARTED")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Context Chunks: {len(retrieval_results)}")
        print(f"{'='*80}\n")
        print("Response: ", end='', flush=True)
        
        # Generate streaming response
        try:
            chunk_count = 0
            for chunk in self.openai_client.chat_completion_stream(messages=messages):
                chunk_count += 1
                print(chunk, end='', flush=True)  # Print to terminal in real-time
                yield chunk
            
            elapsed = time.time() - start_time
            
            # Update statistics (token usage not available in streaming)
            self.total_generations += 1
            self.total_generation_time += elapsed
            
            # Print footer to terminal
            print(f"\n\n{'='*80}")
            print(f"STREAMING COMPLETE")
            print(f"{'='*80}")
            print(f"Chunks Received: {chunk_count}")
            print(f"Generation Time: {elapsed:.2f}s")
            print(f"Note: Token usage not available in streaming mode")
            print(f"{'='*80}\n")
            
            logger.info(
                f"Streaming response complete | "
                f"chunks={chunk_count} | "
                f"time={elapsed:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}", exc_info=True)
            raise
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        avg_tokens = (
            self.total_tokens_used / self.total_generations
            if self.total_generations > 0
            else 0
        )
        avg_time = (
            self.total_generation_time / self.total_generations
            if self.total_generations > 0
            else 0
        )
        
        return {
            "total_generations": self.total_generations,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_generation": avg_tokens,
            "total_generation_time": self.total_generation_time,
            "avg_generation_time": avg_time,
        }


# =============================================================================
# Conversation History Management
# =============================================================================

class ConversationMemory:
    """
    Manages conversation history for multi-turn conversations.
    Uses a native Python list to store OpenAI-formatted messages.
    """
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to keep (excluding system)
        """
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
        logger.info(f"ConversationMemory initialized | max_messages={max_messages}")
    
    def add_user_message(self, content: str):
        """Add user message to history."""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()
        logger.debug(f"User message added: {content[:50]}...")
    
    def add_assistant_message(self, content: str):
        """Add assistant message to history."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()
        logger.debug(f"Assistant message added: {content[:50]}...")
        
    def _trim_history(self):
        """Keep the memory within max_messages bounds."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history in OpenAI format."""
        return list(self.messages)
    
    def clear(self):
        """Clear conversation history."""
        self.messages.clear()
        logger.info("Conversation history cleared")


# =============================================================================
# Singleton
# =============================================================================

_generator_service_instance = None


def get_generator_service() -> GeneratorService:
    """
    Get singleton generator service instance.
    
    Returns:
        GeneratorService instance
    """
    global _generator_service_instance
    
    if _generator_service_instance is None:
        _generator_service_instance = GeneratorService()
    
    return _generator_service_instance