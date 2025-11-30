import base64
import numpy as np
import cv2
from typing import Optional
from app.config import settings

def decode_image(image_data: str) -> Optional[np.ndarray]:
    """Decode base64 image to numpy array"""
    try:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for face recognition"""
    # Resize if too large
    height, width = image.shape[:2]
    max_size = settings.IMAGE_MAX_SIZE
    
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def detect_face(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect face in image and return cropped face"""
    try:
        # Ensure image is in correct format
        if image is None or image.size == 0:
            print("Invalid image provided")
            return None
        
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Load Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("Failed to load Haar Cascade classifier")
            return None
        
        # Try multiple detection passes with different parameters
        faces = None
        
        # First attempt: Standard parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Second attempt: More lenient parameters
        if len(faces) == 0:
            print("First detection failed, trying with relaxed parameters...")
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Third attempt: Even more lenient
        if len(faces) == 0:
            print("Second detection failed, trying with very relaxed parameters...")
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        # If still no face detected, return None (DO NOT use fallback)
        if len(faces) == 0:
            print("❌ No face detected after multiple attempts")
            return None
        
        # If multiple faces detected, use the largest one
        if len(faces) > 1:
            # Sort by area (width * height) and take the largest
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            print(f"Multiple faces detected ({len(faces)}), using largest face")
        
        # Extract the face region
        x, y, w, h = faces[0]
        print(f"Face detected at position: x={x}, y={y}, w={w}, h={h}")
        
        # Add padding around the face (10% on each side)
        padding = int(0.1 * max(w, h))
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        # Crop the face region
        face_image = image[y_start:y_end, x_start:x_end]
        
        return face_image
        
    except Exception as e:
        print(f"Error detecting face: {e}")
        import traceback
        traceback.print_exc()
        return None
