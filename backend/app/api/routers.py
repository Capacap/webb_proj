from fastapi import APIRouter
from app.api.endpoints.authentication import router as auth_router
from app.api.endpoints.textgen import router as textgen_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(textgen_router)
