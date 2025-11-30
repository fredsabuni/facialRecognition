import numpy as np
import cv2
from typing import Optional, Tuple
from app.faiss_index import faiss_index
from app.utils.image_utils import decode_image, preprocess_image
from app.utils.liveness import check_liveness
from app.config import settings
import face_recognition

class FaceService:
    def __init__(self):
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
    
    def extract_embedding(self, image_data: str) -> Optional[np.ndarray]:
        """Extract face embedding from image"""
        from app.utils.image_utils import detect_face
        
        print("Starting embedding extraction...")
        
        image = decode_image(image_data)
        if image is None:
            print("Failed to decode image")
            return None
        
        print(f"Image decoded successfully. Shape: {image.shape}")
        
        processed = preprocess_image(image)
        print(f"Image preprocessed. Shape: {processed.shape}")
        
        # Detect face in the image
        face = detect_face(processed)
        if face is None or face.size == 0:
            print("No face detected in image")
            return None
        
        print(f"Face detected successfully. Shape: {face.shape}")
        
        try:
            # Use face_recognition to extract embedding from the detected face
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(face_rgb)
            if encodings:
                embedding = encodings[0].astype('float32')
                print(f"Embedding created successfully. Shape: {embedding.shape}")
                return embedding
            else:
                print("No face encoding found")
                return None
        except Exception as e:
            print(f"Error creating embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def enroll_face(self, user_id: str, image_data: str) -> bool:
        """Enroll a new face"""
        embedding = self.extract_embedding(image_data)
        if embedding is None:
            return False
        
        faiss_index.add_embedding(user_id, embedding)
        return True
    
    def verify_face(self, user_id: str, image_data: str) -> Tuple[bool, float, bool]:
        """Verify if image matches enrolled user"""
        embedding = self.extract_embedding(image_data)
        if embedding is None:
            return False, 0.0, False
        
        results = faiss_index.search(embedding, k=1)
        if not results:
            return False, 0.0, False
        
        matched_user, distance = results[0]
        confidence = 1.0 / (1.0 + distance)
        verified = matched_user == user_id and confidence >= self.similarity_threshold
        
        image = decode_image(image_data)
        liveness_passed = check_liveness(image)
        
        return verified, confidence, liveness_passed
    
    def identify_face(self, image_data: str) -> Optional[Tuple[str, float]]:
        """Identify face from database"""
        embedding = self.extract_embedding(image_data)
        if embedding is None:
            return None
        
        results = faiss_index.search(embedding, k=1)
        if not results:
            return None
        
        user_id, distance = results[0]
        confidence = 1.0 / (1.0 + distance)
        
        if confidence >= self.similarity_threshold:
            return user_id, confidence
        return None

face_service = FaceService()
