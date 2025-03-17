from fastapi import APIRouter

router = APIRouter(tags=["auth"], prefix="/auth")

@router.get("/auth", status_code=200)
def login():
    return True