from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.schemas import (
    FaceEnrollRequest, FaceVerifyRequest, FaceIdentifyRequest,
    EnrollResponse, VerifyResponse, FaceResponse
)
from app.face_service import face_service
from app.faiss_index import faiss_index
from app.database import get_db
from app.models import FaceRecord
import base64

router = APIRouter(prefix="/recognition", tags=["recognition"])

@router.post("/enroll", response_model=EnrollResponse)
async def enroll_face(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Enroll a new face with fraud prevention.
    
    SECURITY: Prevents the same face from being enrolled with multiple user IDs.
    """
    # Check if user ID already exists
    existing = db.query(FaceRecord).filter(FaceRecord.user_id == user_id).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"User ID '{user_id}' is already enrolled"
        )
    
    # Read file and convert to base64
    contents = await file.read()
    base64_string = base64.b64encode(contents).decode('utf-8')
    
    # Extract face embedding
    embedding = face_service.extract_embedding(base64_string)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear photo showing your face."
        )
    
    # CRITICAL: Check if this face already exists in the system
    results = faiss_index.search(embedding, k=1)
    
    if results and len(results) > 0:
        matched_user_id, distance = results[0]
        confidence = 1.0 / (1.0 + distance)
        
        # If confidence is high, this face already exists
        if confidence >= face_service.similarity_threshold:
            raise HTTPException(
                status_code=403,
                detail=f"FRAUD ALERT: This face is already registered to user '{matched_user_id}'. Cannot enroll the same face with multiple user IDs."
            )
    
    # Face is unique - proceed with enrollment
    success = face_service.enroll_face(user_id, base64_string)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to enroll face. Please try again."
        )
    
    face_record = FaceRecord(user_id=user_id)
    db.add(face_record)
    db.commit()
    
    return EnrollResponse(
        user_id=user_id,
        message=f"User '{user_id}' enrolled successfully",
        success=True
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_face(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify face with fraud prevention.
    
    SECURITY: Checks if the face already exists in the system to prevent
    the same person from enrolling with multiple user IDs.
    """
    # Read file and convert to base64
    contents = await file.read()
    base64_string = base64.b64encode(contents).decode('utf-8')
    
    # Extract face embedding from the uploaded image
    from app.utils.image_utils import decode_image
    from app.utils.liveness import check_liveness
    
    embedding = face_service.extract_embedding(base64_string)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear photo showing your face."
        )
    
    # Check liveness
    image = decode_image(base64_string)
    liveness_passed = check_liveness(image) if image is not None else False
    
    # CRITICAL: Search if this face already exists in the system
    results = faiss_index.search(embedding, k=1)
    
    if results and len(results) > 0:
        # Face found in database
        matched_user_id, distance = results[0]
        confidence = 1.0 / (1.0 + distance)
        
        # Check if confidence is high enough to consider it a match
        if confidence >= face_service.similarity_threshold:
            # This face already exists in the system
            if matched_user_id == user_id:
                # Same user, same face - VERIFIED
                return VerifyResponse(
                    verified=True,
                    confidence=confidence,
                    liveness_passed=liveness_passed,
                    auto_enrolled=False,
                    user_exists=True,
                    message=f"Face verified successfully for user {user_id}"
                )
            else:
                # FRAUD ATTEMPT: Same face, different user ID!
                raise HTTPException(
                    status_code=403,
                    detail=f"FRAUD ALERT: This face is already registered to user '{matched_user_id}'. Cannot enroll the same face with multiple user IDs."
                )
    
    # Face not found in system - check if user_id exists
    existing_user = db.query(FaceRecord).filter(FaceRecord.user_id == user_id).first()
    
    if existing_user:
        # User ID exists but face doesn't match
        raise HTTPException(
            status_code=403,
            detail=f"User ID '{user_id}' already exists with a different face. Cannot replace existing user's face."
        )
    
    # New face, new user - ENROLL
    success = face_service.enroll_face(user_id, base64_string)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to enroll face. Please try again."
        )
    
    face_record = FaceRecord(user_id=user_id)
    db.add(face_record)
    db.commit()
    
    return VerifyResponse(
        verified=True,
        confidence=1.0,
        liveness_passed=liveness_passed,
        auto_enrolled=True,
        user_exists=False,
        message=f"New user '{user_id}' enrolled successfully"
    )

@router.post("/verify-base64", response_model=VerifyResponse)
async def verify_face_base64(request: FaceVerifyRequest, db: Session = Depends(get_db)):
    """Verify face using base64 encoded image. Auto-enrolls if not found."""
    # First check if user exists in database
    existing = db.query(FaceRecord).filter(FaceRecord.user_id == request.user_id).first()
    
    if not existing:
        # User doesn't exist - auto-enroll
        success = face_service.enroll_face(request.user_id, request.image)
        if not success:
            raise HTTPException(
                status_code=400, 
                detail="No face detected in the image. Please upload a clear photo showing your face."
            )
        
        face_record = FaceRecord(user_id=request.user_id)
        db.add(face_record)
        db.commit()
        
        # Check liveness for the enrolled face
        from app.utils.image_utils import decode_image
        from app.utils.liveness import check_liveness
        image = decode_image(request.image)
        liveness_passed = check_liveness(image) if image is not None else False
        
        return VerifyResponse(
            verified=True,
            confidence=1.0,
            liveness_passed=liveness_passed,
            auto_enrolled=True,
            user_exists=False,
            message="New user enrolled successfully"
        )
    
    # User exists - verify face
    verified, confidence, liveness_passed = face_service.verify_face(
        request.user_id, request.image
    )
    
    if not verified:
        raise HTTPException(
            status_code=401,
            detail=f"Face verification failed. Confidence: {confidence:.2f}. User exists but face doesn't match."
        )
    
    return VerifyResponse(
        verified=verified,
        confidence=confidence,
        liveness_passed=liveness_passed,
        auto_enrolled=False,
        user_exists=True,
        message="Face verified successfully"
    )

@router.post("/identify", response_model=FaceResponse)
async def identify_face(file: UploadFile = File(...)):
    """Identify a face from the database by uploading an image file"""
    # Read file and convert to base64
    contents = await file.read()
    base64_string = base64.b64encode(contents).decode('utf-8')
    
    result = face_service.identify_face(base64_string)
    if result is None:
        raise HTTPException(status_code=404, detail="No matching face found")
    
    user_id, confidence = result
    return FaceResponse(user_id=user_id, confidence=confidence)

@router.post("/identify-base64", response_model=FaceResponse)
async def identify_face_base64(request: FaceIdentifyRequest):
    """Identify a face from the database using base64 encoded image"""
    result = face_service.identify_face(request.image)
    if result is None:
        raise HTTPException(status_code=404, detail="No matching face found")
    
    user_id, confidence = result
    return FaceResponse(user_id=user_id, confidence=confidence)

 