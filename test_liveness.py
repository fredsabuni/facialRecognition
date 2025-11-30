"""
Test script for liveness detection
"""
import cv2
import numpy as np
from app.utils.liveness import check_liveness, calculate_liveness_score, detect_spoofing

def test_liveness_detection():
    """Test liveness detection with sample images"""
    
    # Test 1: Create a synthetic "live" image with good texture
    print("Test 1: High-quality image (simulated live face)")
    live_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add some texture variation
    live_image = cv2.GaussianBlur(live_image, (5, 5), 0)
    
    is_live = check_liveness(live_image)
    score = calculate_liveness_score(live_image)
    is_spoof, spoof_confidence = detect_spoofing(live_image)
    
    print(f"  Is Live: {is_live}")
    print(f"  Liveness Score: {score:.3f}")
    print(f"  Is Spoof: {is_spoof} (confidence: {spoof_confidence:.3f})")
    print()
    
    # Test 2: Create a synthetic "spoof" image (flat, low texture)
    print("Test 2: Low-quality flat image (simulated printed photo)")
    spoof_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Add minimal variation
    spoof_image += np.random.randint(-5, 5, (480, 640, 3), dtype=np.int8)
    spoof_image = np.clip(spoof_image, 0, 255).astype(np.uint8)
    
    is_live = check_liveness(spoof_image)
    score = calculate_liveness_score(spoof_image)
    is_spoof, spoof_confidence = detect_spoofing(spoof_image)
    
    print(f"  Is Live: {is_live}")
    print(f"  Liveness Score: {score:.3f}")
    print(f"  Is Spoof: {is_spoof} (confidence: {spoof_confidence:.3f})")
    print()
    
    # Test 3: Empty/invalid image
    print("Test 3: Invalid image")
    is_live = check_liveness(None)
    print(f"  Is Live: {is_live}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Liveness Detection Test")
    print("=" * 60)
    print()
    test_liveness_detection()
    print("=" * 60)
    print("Test completed!")
