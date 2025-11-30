 #!/usr/bin/env python3
"""Test script to verify image processing"""

import base64
import sys
from app.utils.image_utils import decode_image, preprocess_image, detect_face
import cv2

def test_image_from_file(image_path):
    """Test loading and processing an image file"""
    print(f"Testing image from file: {image_path}")
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"Base64 string length: {len(base64_string)}")
    print(f"First 50 chars: {base64_string[:50]}")
    
    # Test decoding
    image = decode_image(base64_string)
    if image is None:
        print("❌ Failed to decode image")
        return False
    
    print(f"✓ Image decoded. Shape: {image.shape}")
    
    # Test preprocessing
    processed = preprocess_image(image)
    print(f"✓ Image preprocessed. Shape: {processed.shape}")
    
    # Test face detection
    face = detect_face(processed)
    if face is None or face.size == 0:
        print("❌ No face detected")
        return False
    
    print(f"✓ Face detected! Shape: {face.shape}")
    
    # Save the detected face for visual verification
    output_path = "detected_face.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    print(f"✓ Detected face saved to: {output_path}")
    
    return True

def test_base64_string(base64_string):
    """Test processing a base64 string directly"""
    print("Testing base64 string...")
    print(f"Base64 string length: {len(base64_string)}")
    
    # Test decoding
    image = decode_image(base64_string)
    if image is None:
        print("❌ Failed to decode image")
        return False
    
    print(f"✓ Image decoded. Shape: {image.shape}")
    
    # Test preprocessing
    processed = preprocess_image(image)
    print(f"✓ Image preprocessed. Shape: {processed.shape}")
    
    # Test face detection
    face = detect_face(processed)
    if face is None or face.size == 0:
        print("❌ No face detected")
        return False
    
    print(f"✓ Face detected! Shape: {face.shape}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_image.py <image_file>")
        print("  python test_image.py --base64 <base64_string>")
        sys.exit(1)
    
    if sys.argv[1] == "--base64":
        if len(sys.argv) < 3:
            print("Please provide base64 string")
            sys.exit(1)
        success = test_base64_string(sys.argv[2])
    else:
        success = test_image_from_file(sys.argv[1])
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
