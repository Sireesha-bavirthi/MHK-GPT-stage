"""
__Project__: Company Chatbot
__Description__: Query reformulation service for conversational RAG. Reformulates vague follow-up queries into standalone queries using conversation history.
__Created Date__: 05-02-2026
__Updated Date__: 06-02-2026
__Author__: SWATHI KAVITI
__Employee Id__: 800340

"""

import logging
from typing import List, Dict, Optional

from app.services.llm.openai_client import get_openai_client, OpenAIClient

logger = logging.getLogger(__name__)


# =============================================================================
# Query Reformulation Service
# =============================================================================

class QueryReformulator:
    """
    Reformulates user queries using conversation history.
    
    Converts vague follow-up questions (e.g., "tell me more about it")
    into standalone queries (e.g., "tell me more about MHK's cloud solutions").
    """
    
    REFORMULATION_PROMPT = """You are a query reformulation assistant. Your job is to rewrite user queries to make them standalone and context-aware.

Given a conversation history and a new user query, rewrite the query to be self-contained by incorporating relevant context from the conversation history.

Rules:
1. If the query is already clear and standalone, return it as-is
2. If the query contains pronouns (it, them, that, this, etc.) or is vague, rewrite it with specific context
3. Keep the reformulated query concise and natural
4. Maintain the user's intent and question type
5. Only output the reformulated query, nothing else

Examples:

Conversation:
User: What cloud solutions does MHK offer?
Assistant: MHK offers AWS, Azure, and Google Cloud solutions...

New Query: explain more about them
Reformulated: Explain more about MHK's cloud solutions

---

Conversation:
User: What is AI?
Assistant: AI is artificial intelligence...

New Query: tell more about it
Reformulated: Tell more about artificial intelligence

---

Conversation:
User: What services does MHK provide?
Assistant: MHK provides cloud, AI, and data engineering services...

New Query: what are the benefits?
Reformulated: What are the benefits of MHK's services?

Now reformulate this query:"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        """
        Initialize query reformulator.
        
        Args:
            openai_client: OpenAI client instance (default: singleton)
        """
        self.openai_client = openai_client or get_openai_client()
        logger.info("QueryReformulator initialized")
    
    def reformulate(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Reformulate a query using conversation history.
        
        Args:
            query: User's original query
            conversation_history: Previous conversation messages
            
        Returns:
            Reformulated query (or original if no history or reformulation fails)
        """
        # If no conversation history, return original query
        if not conversation_history or len(conversation_history) == 0:
            logger.debug("No conversation history, using original query")
            return query
        
        # If query is already long/detailed, probably doesn't need reformulation
        if len(query.split()) > 10:
            logger.debug("Query is detailed enough, using original")
            return query
        
        try:
            # Build conversation context
            context_parts = []
            for msg in conversation_history[-4:]:  # Use last 4 messages (2 turns)
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    # Truncate long assistant responses
                    truncated = content[:200] + "..." if len(content) > 200 else content
                    context_parts.append(f"Assistant: {truncated}")
            
            conversation_context = "\n".join(context_parts)
            
            # Build reformulation prompt
            full_prompt = f"""{self.REFORMULATION_PROMPT}

Conversation:
{conversation_context}

New Query: {query}
Reformulated:"""
            
            # Call OpenAI to reformulate
            messages = [
                {"role": "system", "content": "You are a helpful query reformulation assistant."},
                {"role": "user", "content": full_prompt}
            ]
            
            logger.debug(f"Reformulating query: '{query}'")
            
            response = self.openai_client.chat_completion(
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent reformulation
                max_tokens=100
            )
            
            reformulated = response.choices[0].message.content.strip()
            
            # Sanity check: if reformulated is too different or empty, use original
            if not reformulated or len(reformulated) < 3:
                logger.warning("Reformulation failed, using original query")
                return query
            
            logger.info(f"Query reformulated: '{query}' -> '{reformulated}'")
            return reformulated
            
        except Exception as e:
            logger.error(f"Reformulation error: {str(e)}, using original query")
            return query


# =============================================================================
# Singleton
# =============================================================================

_reformulator_instance = None


def get_query_reformulator() -> QueryReformulator:
    """
    Get singleton query reformulator instance.
    
    Returns:
        QueryReformulator instance
    """
    global _reformulator_instance
    
    if _reformulator_instance is None:
        _reformulator_instance = QueryReformulator()
    
    return _reformulator_instance
