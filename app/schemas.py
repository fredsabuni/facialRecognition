from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class FaceEnrollRequest(BaseModel):
    user_id: str
    image: str  # base64 encoded image

class FaceVerifyRequest(BaseModel):
    user_id: str
    image: str  # base64 encoded image

class FaceIdentifyRequest(BaseModel):
    image: str  # base64 encoded image

class FaceResponse(BaseModel):
    user_id: str
    confidence: float
    liveness_score: Optional[float] = None

class EnrollResponse(BaseModel):
    user_id: str
    message: str
    success: bool

class VerifyResponse(BaseModel):
    verified: bool
    confidence: float
    liveness_passed: bool
    auto_enrolled: bool = False
    user_exists: bool = True
    message: Optional[str] = None
