from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, universal_restoration

app = FastAPI(title="GenAI Assignment 1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(universal_restoration.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "GenAI Assignment 1 API is running"}
