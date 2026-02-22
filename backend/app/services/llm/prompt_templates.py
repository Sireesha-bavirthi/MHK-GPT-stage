"""
Prompt templates for RAG chatbot.
Manages system and user prompts with context formatting.
"""

from typing import List, Dict, Any
from app.services.rag.retriever import RetrievalResult


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant for {company_name}.
You are MHK Nova, AI assistant of MHK Tech Inc
Use "we" and "our" when referring to MHK — you're part of the team
IMPORTANT RULES:
 1. **Casual conversation, Confirmations & Persona** (greetings, closings like "bye" or "thank you", "how are you", "ok", "sure", "no thanks", your identity, basic company info):
   - Respond naturally like a helpful human assistant. If the user says "ok", "thanks" or "no thanks", acknowledge it politely and ask if there's anything else you can help with.
   - Be warm, friendly, and highly conversational.
   - Example (greeting): "I'm doing well, thank you! How can I assist you today?"
   - Example (closing): "You're very welcome! Have a great day!" or "Goodbye! Feel free to reach out if you need anything else."
   - Don't mention you're an AI unless specifically asked.
   - Your name is MHK Nova, and you are a helpful AI assistant built for {company_name}.
   - Your core capabilities are: answering questions about {company_name}, scheduling meetings with our team, and helping users find and apply for open job positions.
   - CRITICAL: ONLY list your capabilities if the user EXPLICITLY asks "what can you do?", "what are your capabilities?", or "who are you?".
   - CRITICAL: DO NOT list your capabilities for simple greetings (e.g., "how are you?", "hi", "hello"). Just respond to the greeting conversationally.
   - Mr. Rajesh is the CEO of the company. (Answer questions about the CEO using this info directly).
   - FORMATTING RULE: Do not output walls of plain text. Use Markdown extensively to make your responses easy to read. Use bullet points (`-`) for lists, and use **bold** text to highlight at least two or three important keywords, names, or concepts in every response.

2. **Factual/informational questions (excluding persona/chitchat above)**: ONLY use the Context below.
   - CRITICAL: If the answer is NOT explicitly stated in the provided Context, you MUST reply ONLY with: "I don't have that information. Please ask about {company_name}'s services or products."
   - DO NOT try to be helpful by providing general definitions or outside knowledge (e.g. if asked about "debugging code" and it's not in the context, do not explain what debugging is).
   - Never make assumptions or go beyond the information provided in the context.

3. **When using Context**: Be clear, concise, and cite sources.

Context:
{context}"""


USER_PROMPT = """Question: {query}

Please provide a helpful, but STRICTLY CONCISE answer based ONLY on the context provided.
Do not elaborate, hallucinate extra details, or rewrite short facts into long paragraphs. Keep your answer brief and directly to the point."""


# =============================================================================
# Context Formatting
# =============================================================================

def format_context_from_results(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results into readable context for the LLM.
    
    Args:
        results: List of RetrievalResult objects from retriever
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant context found."
    
    context_parts = []
    
    for idx, result in enumerate(results, 1):
        file_name = result.metadata.get('file_name', 'Unknown')
        chunk_index = result.metadata.get('chunk_index', 'N/A')
        score = result.score
        
        # Format each document chunk
        context_part = f"""Document {idx} (Source: {file_name}, Chunk: {chunk_index}, Relevance: {score:.3f}):
{result.text}"""
        
        context_parts.append(context_part)
    
    # Join all parts with separator
    return "\n\n" + ("-" * 80) + "\n\n".join(context_parts)


def format_system_prompt(company_name: str, context: str) -> str:
    """
    Format system prompt with company name and context.
    
    Args:
        company_name: Name of the company
        context: Formatted context from retrieval results
        
    Returns:
        Formatted system prompt
    """
    return SYSTEM_PROMPT.format(
        company_name=company_name,
        context=context
    )


def format_user_prompt(query: str) -> str:
    """
    Format user prompt with query.
    
    Args:
        query: User's question
        
    Returns:
        Formatted user prompt
    """
    return USER_PROMPT.format(query=query)


# =============================================================================
# Conversation History Formatting
# =============================================================================

def build_messages(
    system_prompt: str,
    user_prompt: str,
    conversation_history: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Build complete message list for OpenAI API.
    
    Args:
        system_prompt: Formatted system prompt
        user_prompt: Formatted user prompt
        conversation_history: Previous messages (optional)
        
    Returns:
        List of message dicts in OpenAI format
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user query
    messages.append({"role": "user", "content": user_prompt})
    
    return messages


def truncate_conversation_history(
    history: List[Dict[str, str]],
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Truncate conversation history to keep only recent messages.
    Always keeps system message.
    
    Args:
        history: Full conversation history
        max_messages: Maximum number of messages to keep (excluding system)
        
    Returns:
        Truncated history
    """
    if not history:
        return []
    
    # Separate system message from rest
    system_msg = None
    other_msgs = []
    
    for msg in history:
        if msg.get("role") == "system":
            system_msg = msg
        elif "role" in msg:
            other_msgs.append(msg)
        else:
            # Skip malformed messages
            continue
    
    # Keep only last N messages
    if len(other_msgs) > max_messages:
        other_msgs = other_msgs[-max_messages:]
    
    # Rebuild with system message first
    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(other_msgs)
    
    return result
