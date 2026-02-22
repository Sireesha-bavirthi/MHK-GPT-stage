"""
__Project__: Company Chatbot
__Description__: OpenAI API client wrapper. Handles chat completions with streaming and token tracking.
__Created Date__: 05-02-2026
__Updated Date__: 06-02-2026
__Author__: SWATHI KAVITI
__Employee Id__: 800340

"""

import logging
import time
from typing import List, Dict, Any, Iterator, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from app.core.config import settings
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Wrapper for OpenAI API client.
    Handles chat completions with streaming and token tracking.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (default from settings)
            model: Model name (default from settings)
            temperature: Sampling temperature (default from settings)
            max_tokens: Max tokens in response (default from settings)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else settings.OPENAI_TEMPERATURE
        self.max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(
                f"OpenAI client initialized | model={self.model} | "
                f"temperature={self.temperature} | max_tokens={self.max_tokens}"
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> ChatCompletion:
        """
        Generate chat completion (non-streaming).
         
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Must be False for this method
            
        Returns:
            ChatCompletion object
            
        Raises:
            LLMError: If API call fails
        """
        if stream:
            raise ValueError("Use chat_completion_stream() for streaming responses")
        
        start_time = time.time()
        
        try:
            logger.info(
                f"Generating chat completion | model={self.model} | "
                f"messages={len(messages)} | stream=False"
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=False
            )
            
            elapsed = time.time() - start_time
            
            # Log token usage
            usage = response.usage
            logger.info(
                f"Chat completion generated | "
                f"prompt_tokens={usage.prompt_tokens} | "
                f"completion_tokens={usage.completion_tokens} | "
                f"total_tokens={usage.total_tokens} | "
                f"time={elapsed:.2f}s"
            )
            
            logger.debug(f"Response content: {response.choices[0].message.content[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}", exc_info=True)
            raise LLMError(f"Failed to generate chat completion: {str(e)}")
    
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        Generate streaming chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Content chunks as they arrive
            
        Raises:
            LLMError: If API call fails
        """
        start_time = time.time()
        total_chunks = 0
        
        try:
            logger.info(
                f"Generating streaming chat completion | model={self.model} | "
                f"messages={len(messages)} | stream=True"
            )
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True
            )
            
            # Stream response chunks
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    total_chunks += 1
                    yield content
            
            elapsed = time.time() - start_time
            
            logger.info(
                f"Streaming completion finished | "
                f"chunks={total_chunks} | "
                f"time={elapsed:.2f}s"
            )
            
            # Note: Token usage not available in streaming mode
            logger.debug("Token usage not available in streaming mode")
            
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {str(e)}", exc_info=True)
            raise LLMError(f"Failed to generate streaming chat completion: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model configuration.
        
        Returns:
            Dictionary with model settings
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# =============================================================================
# Singleton
# =============================================================================

_openai_client_instance = None


def get_openai_client() -> OpenAIClient:
    """
    Get singleton OpenAI client instance.
    
    Returns:
        OpenAIClient instance
    """
    global _openai_client_instance
    
    if _openai_client_instance is None:
        _openai_client_instance = OpenAIClient()
    
    return _openai_client_instance
