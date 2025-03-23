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

load_dotenv()  # Load environment variables

router = APIRouter(tags=["textgen"], prefix="/textgen")

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]  # Expecting format: [{"role": "user", "content": "..."}]
    conversation_id: int | None = None  # Add conversation reference

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

@router.post("/generate")
async def generate_text(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")

    # Get or create conversation
    conversation = None
    if request.conversation_id:
        conversation = db.get(Conversation, request.conversation_id)
        if not conversation or conversation.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # Create new conversation if none exists
    if not conversation:
        conversation_title = request.messages[0]['content'][:50]  # Truncate first message for title
        conversation = Conversation(
            title=conversation_title,
            user_id=current_user.id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Store user message
    user_message = None
    if request.messages:
        last_user_msg = next(m for m in reversed(request.messages) if m['role'] == 'user')
        user_message = ConversationMessage(
            text=last_user_msg['content'],
            is_user=True,
            conversation_id=conversation.id
        )
        db.add(user_message)

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=request.messages,
        )
        
        # Store AI response
        ai_message = ConversationMessage(
            text=response.choices[0].message.content,
            is_user=False,
            conversation_id=conversation.id
        )
        db.add(ai_message)
        db.commit()

        return {
            "content": ai_message.text,
            "conversation_id": conversation.id,
            "message_id": ai_message.id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations", response_model=list[ConversationOut])
def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all conversations for current user"""
    conversations = db.execute(
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

@router.get("/conversations/{conversation_id}", response_model=list[MessageOut])
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all messages in a specific conversation"""
    conversation = db.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return [{
        "id": m.id,
        "text": m.text,
        "is_user": m.is_user,
        "created_at": m.created_at
    } for m in conversation.messages]