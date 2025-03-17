import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

router = APIRouter(tags=["textgen"], prefix="/textgen")

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]  # Expecting format: [{"role": "user", "content": "..."}]

@router.post("/generate")
async def generate_text(request: ChatRequest):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")

    client = Mistral(api_key=api_key)
    
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=request.messages,
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.model_dump()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))