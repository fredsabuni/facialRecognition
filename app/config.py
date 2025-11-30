from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./faces.db"
    FAISS_INDEX_PATH: str = "./faiss_index.bin"
    FACE_DETECTION_MODEL: str = "retinaface"
    FACE_RECOGNITION_MODEL: str = "facenet"
    SIMILARITY_THRESHOLD: float = 0.6
    IMAGE_MAX_SIZE: int = 2048
    
    class Config:
        env_file = ".env"

settings = Settings()
