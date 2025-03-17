from fastapi import APIRouter

router = APIRouter(tags=["textgen"], prefix="/textgen")

@router.get("/gen", status_code=200)
def gen():
    return True