import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from mistralai import Mistral
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.db_setup import get_db
from app.api.models import Conversation, ConversationMessage, User
from app.security import get_current_user
from datetime import datetime
from sqlalchemy import select
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import glob
import shutil

from app.core.vector_db import VectorStore, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
LOGS_DIR = Path(os.path.dirname(__file__)).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Maximum number of log files to keep
MAX_LOG_FILES = 100

def rotate_logs() -> None:
    """
    Rotate log files to maintain a maximum number of entries.
    """
    try:
        # Get all log files
        log_files = sorted(LOGS_DIR.glob("ai_interactions_*.json"))
        
        # If we exceed the maximum, remove oldest files
        if len(log_files) > MAX_LOG_FILES:
            # Calculate how many files to remove
            files_to_remove = len(log_files) - MAX_LOG_FILES
            
            # Remove oldest files
            for file in log_files[:files_to_remove]:
                try:
                    file.unlink()
                    logger.info(f"Removed old log file: {file}")
                except Exception as e:
                    logger.error(f"Failed to remove old log file {file}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error during log rotation: {str(e)}")

# Path to the vector database directory
VECTOR_DB_PATH: str = os.path.join(os.path.dirname(__file__), "../../../resources/vector_db")

load_dotenv()  # Load environment variables

router = APIRouter(tags=["textgen"], prefix="/textgen")

# Initialize the vector database - load only once when the API starts
vector_db = None
try:
    logger.info(f"Loading vector database from {VECTOR_DB_PATH}")
    vector_db = VectorStore(vector_db_path=VECTOR_DB_PATH)
    logger.info("Vector database loaded successfully")
except Exception as e:
    logger.error(f"Failed to load vector database: {str(e)}")
    logger.warning("RAG functionality will be disabled")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # Expecting format: [{"role": "user", "content": "..."}]
    conversation_id: Optional[int] = None  # Add conversation reference
    use_rag: bool = True  # Option to enable/disable RAG

class ChatResponse(BaseModel):
    content: str
    conversation_id: int
    message_id: int

class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    message_count: int

class MessageOut(BaseModel):
    id: int
    text: str
    is_user: bool
    created_at: datetime

def get_relevant_context(query: str, messages: List[Dict[str, str]], top_k: int = 3) -> str:
    """
    Retrieve relevant context from the vector database based on the user query and conversation history.
    
    Args:
        query: The user's current message
        messages: The full conversation history
        top_k: Number of relevant chunks to retrieve
        
    Returns:
        A string containing the relevant context information
    """
    if vector_db is None:
        return ""
    
    try:
        # Extract relevant context from conversation history
        conversation_context = []
        for msg in reversed(messages[:-1]):  # Exclude the current message
            if msg["role"] == "user":
                conversation_context.append(msg["content"])
        
        # Combine current query with conversation context
        full_context = " ".join([*conversation_context, query])
        
        # First search: Get broad context
        broad_results = vector_db.search(full_context, k=top_k * 2)
        
        # Second search: Get specific context about the current query
        specific_results = vector_db.search(query, k=top_k)
        
        # Combine and deduplicate results
        all_results = []
        seen_titles = set()
        
        # Add specific results first (they're more relevant to the current query)
        for result in specific_results:
            title = result["metadata"].get("title", "Unknown")
            if title not in seen_titles:
                seen_titles.add(title)
                all_results.append(result)
        
        # Add broad results that aren't duplicates
        for result in broad_results:
            title = result["metadata"].get("title", "Unknown")
            if title not in seen_titles:
                seen_titles.add(title)
                all_results.append(result)
        
        if not all_results:
            return ""
        
        # Format the results
        context_parts: List[str] = []
        
        for result in all_results[:top_k]:  # Take top k results after deduplication
            title: str = result["metadata"].get("title", "Unknown")
            content: str = result["metadata"].get("content_preview", "").replace("...", "")
            score: float = result.get("score", 0.0)
            
            # Format the content with better structure
            context_parts.append(
                f"Information from '{title}':\n"
                f"{content}\n"
                f"(Relevance score: {score:.2f})\n"
            )
        
        return "\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return ""

def log_ai_interaction(
    user_query: str,
    context: str,
    messages: List[Dict[str, str]],
    response: str,
    conversation_id: int,
    use_rag: bool
) -> None:
    """
    Log AI interaction details to a JSON file.
    
    Args:
        user_query: The user's input query
        context: The RAG context used
        messages: The full message history sent to the AI
        response: The AI's response
        conversation_id: The ID of the conversation
        use_rag: Whether RAG was enabled
    """
    # Rotate logs before creating new entry
    rotate_logs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"ai_interactions_{timestamp}.json"
    
    # Extract system message and RAG context
    system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
    
    # Format the log data with more detailed information
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "use_rag": use_rag,
        "user_query": user_query,
        "system_message": system_message,
        "rag_context": {
            "raw_context": context,
            "context_sources": context.split("\n\n") if context else []
        },
        "message_history": [
            {
                "role": msg["role"],
                "content": msg["content"]
            } for msg in messages
        ],
        "ai_response": response,
        "model_config": {
            "model": "mistral-large-latest",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
    }
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Logged AI interaction to {log_file}")
    except Exception as e:
        logger.error(f"Failed to log AI interaction: {str(e)}")

@router.post("/generate", response_model=ChatResponse)
async def generate_text(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    api_key: Optional[str] = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")

    # Get or create conversation
    conversation: Optional[Conversation] = None
    if request.conversation_id:
        conversation = db.get(Conversation, request.conversation_id)
        if not conversation or conversation.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # Create new conversation if none exists
    if not conversation:
        conversation_title: str = request.messages[0]['content'][:50]  # Truncate first message for title
        conversation = Conversation(
            title=conversation_title,
            user_id=current_user.id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Store user message
    user_message: Optional[ConversationMessage] = None
    user_query: str = ""
    if request.messages:
        last_user_msg: Dict[str, str] = next(m for m in reversed(request.messages) if m['role'] == 'user')
        user_query = last_user_msg['content']
        user_message = ConversationMessage(
            text=user_query,
            is_user=True,
            conversation_id=conversation.id
        )
        db.add(user_message)

    try:
        # Get relevant context if RAG is enabled
        context: str = ""
        if request.use_rag and vector_db is not None:
            context = get_relevant_context(user_query, request.messages)
            if context:
                logger.info("Retrieved relevant context for query")
                logger.debug(f"Context: {context}")  # Add debug logging for context
            else:
                logger.info("No relevant context found")
        
        # Prepare messages for the LLM
        messages: List[Dict[str, str]] = request.messages.copy()
        
        # If we have context, add a system message with the context
        if context:
            # Add a system message at the beginning of the conversation
            system_msg: Dict[str, str] = {
                "role": "system",
                "content": (
                    "You are a knowledgeable AI assistant with access to a comprehensive knowledge base. "
                    "Your responses should be accurate, informative, and well-structured. "
                    "Follow these guidelines:\n\n"
                    "1. Use the provided information to answer questions accurately and comprehensively\n"
                    "2. If the information is not relevant to the question, you may ignore it\n"
                    "3. Be precise and specific in your answers, citing sources when possible\n"
                    "4. If you're not certain about something, say so and explain why\n"
                    "5. Maintain a helpful and professional tone\n"
                    "6. If the user's question is unclear, ask for clarification\n"
                    "7. Consider the relevance scores when using information\n"
                    "8. Structure your responses logically with clear paragraphs\n"
                    "9. Include relevant details and examples when appropriate\n"
                    "10. If the information is incomplete, acknowledge the limitations\n\n"
                    f"Here is the relevant information from the knowledge base:\n\n{context}"
                )
            }
            
            # Insert system message at the beginning
            messages.insert(0, system_msg)
        
        # Create Mistral client with context manager
        with Mistral(api_key=api_key) as client:
            # Call the chat completion API with mistral-large
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            
            # Validate response
            if not response.choices or not response.choices[0].message.content:
                raise HTTPException(status_code=500, detail="Invalid response from AI model")
            
            ai_response = response.choices[0].message.content
            
            # Store AI response
            ai_message: ConversationMessage = ConversationMessage(
                text=ai_response,
                is_user=False,
                conversation_id=conversation.id
            )
            db.add(ai_message)
            db.commit()

            # Log the AI interaction
            log_ai_interaction(
                user_query=user_query,
                context=context,
                messages=messages,
                response=ai_response,
                conversation_id=conversation.id,
                use_rag=request.use_rag
            )

            return {
                "content": ai_message.text,
                "conversation_id": conversation.id,
                "message_id": ai_message.id
            }

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations", response_model=List[ConversationOut])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get all conversations for current user"""
    conversations: List[Conversation] = db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
    ).scalars().all()
    
    return [{
        "id": c.id,
        "title": c.title,
        "created_at": c.created_at,
        "message_count": len(c.messages)
    } for c in conversations]

@router.get("/conversations/{conversation_id}", response_model=List[MessageOut])
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get all messages in a specific conversation"""
    conversation: Optional[Conversation] = db.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return [{
        "id": m.id,
        "text": m.text,
        "is_user": m.is_user,
        "created_at": m.created_at
    } for m in conversation.messages]

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """Delete a specific conversation and all its messages."""
    conversation: Optional[Conversation] = db.get(Conversation, conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this conversation")
    
    try:
        # Delete all messages first (cascade should handle this, but being explicit)
        for message in conversation.messages:
            db.delete(message)
        
        # Delete the conversation
        db.delete(conversation)
        db.commit()
        
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete conversation")