from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import textgen, authentication
from app.db_setup import init_db

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(authentication.router)
app.include_router(textgen.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)