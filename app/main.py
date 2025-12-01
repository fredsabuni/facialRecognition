from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import recognition
from app.database import init_db
import os


app = FastAPI(title="Facial Recognition Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the folder where your HTML file is located
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(recognition.router)

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    return {"message": "Facial Recognition Service API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
