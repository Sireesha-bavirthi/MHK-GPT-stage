from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    """
    Single chat message.
    """
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, description="User query", example="What is RAG?")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history tracking")
    history: Optional[List[Message]] = Field(default=[], description="Chat history")
    meeting_session_id: Optional[str] = Field(None, description="Active meeting session ID — must be echoed back each turn")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What services do you offer?",
                "history": [
                    {"role": "user", "content": "Hi there"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"}
                ]
            }
        }

class Source(BaseModel):
    """Source document information."""
    file_name: str
    page_content: str
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    total_duration: float = Field(0.0, description="Processing time in seconds")
    meeting_session_id: Optional[str] = Field(None, description="Active meeting session ID — echo this back in subsequent requests")
