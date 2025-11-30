# Liveness Detection Implementation

## Overview

The liveness detection system uses **passive detection** techniques to determine if a face image comes from a live person or a spoofing attempt (printed photo, screen display, mask, etc.).

## How It Works

The system analyzes a single image using multiple heuristics and combines them into an overall liveness score:

### 1. Texture Analysis (30% weight)
- Uses Laplacian operator to detect texture variation
- **Live faces**: Rich skin texture, pores, fine details (high variance)
- **Spoofed faces**: Flat, uniform texture from printing/screen (low variance)
- Score based on texture variance (normalized to 0-1)

### 2. Color Diversity (25% weight)
- Analyzes color distribution in HSV space
- **Live faces**: Natural color variation, skin tones, shadows
- **Spoofed faces**: Limited color palette, uniform appearance
- Uses entropy calculation on hue and saturation histograms

### 3. Frequency Analysis (25% weight)
- Performs FFT (Fast Fourier Transform) to analyze frequency domain
- **Live faces**: Rich high-frequency content (details, edges)
- **Spoofed faces**: Moiré patterns, pixel grids, limited high frequencies
- Calculates ratio of high-frequency to total energy

### 4. Sharpness Check (20% weight)
- Measures image sharpness using Laplacian variance
- **Live faces**: Sharp, clear details
- **Spoofed faces**: Often blurry from re-capture (photo of photo)

## Usage

### Basic Liveness Check

```python
from app.utils.liveness import check_liveness
import cv2

# Load image
image = cv2.imread("face.jpg")

# Check if live (returns True/False)
is_live = check_liveness(image, threshold=0.5)

if is_live:
    print("Live face detected!")
else:
    print("Possible spoofing attempt!")
```

### Get Detailed Score

```python
from app.utils.liveness import calculate_liveness_score

# Get detailed liveness score (0-1)
score = calculate_liveness_score(image)
print(f"Liveness score: {score:.3f}")

# Interpret score:
# 0.7-1.0: Very likely live
# 0.5-0.7: Probably live
# 0.3-0.5: Uncertain
# 0.0-0.3: Likely spoof
```

### Detect Spoofing

```python
from app.utils.liveness import detect_spoofing

# Detect spoofing with confidence
is_spoof, confidence = detect_spoofing(image)

if is_spoof:
    print(f"Spoofing detected! Confidence: {confidence:.2%}")
```

## Configuration

You can adjust the liveness threshold in your verification logic:

```python
# Strict (fewer false accepts, more false rejects)
is_live = check_liveness(image, threshold=0.7)

# Balanced (default)
is_live = check_liveness(image, threshold=0.5)

# Lenient (more false accepts, fewer false rejects)
is_live = check_liveness(image, threshold=0.3)
```

## Limitations

### Current Implementation
- **Passive detection only**: Analyzes single static image
- **No motion analysis**: Cannot detect blinks, head movement
- **No depth information**: Cannot use 3D structure
- **Heuristic-based**: Not ML-based, may have lower accuracy

### Known Weaknesses
- High-quality prints with good texture may pass
- Professional displays with high resolution may pass
- 3D masks with realistic texture may pass

## Improvements for Production

For production use, consider:

1. **Active Liveness Detection**
   - Challenge-response (blink, smile, turn head)
   - Requires video stream or multiple frames
   - Much more reliable but requires user interaction

2. **Deep Learning Models**
   - Pre-trained models like FaceNet, ArcFace with liveness head
   - Libraries: `face-recognition`, `deepface`, `insightface`
   - Higher accuracy but requires more compute

3. **Multi-Frame Analysis**
   - Analyze video stream for temporal consistency
   - Detect micro-expressions, natural movements
   - More robust against sophisticated attacks

4. **Depth Sensing**
   - Use depth cameras (iPhone Face ID, RealSense)
   - Detect 3D structure of face
   - Very effective against 2D attacks

## Example Integration

The liveness check is already integrated in the verification endpoint:

```python
@router.post("/verify")
async def verify_face(request: FaceVerifyRequest):
    verified, confidence, liveness_passed = face_service.verify_face(
        request.user_id, request.image
    )
    
    return VerifyResponse(
        verified=verified and liveness_passed,  # Both must pass
        confidence=confidence,
        liveness_passed=liveness_passed
    )
```

## Testing

Run the test script to see how it works:

```bash
cd facial-recognition-service
python test_liveness.py
```

## References

- [ISO/IEC 30107-3: Presentation Attack Detection](https://www.iso.org/standard/67381.html)
- [Face Liveness Detection: A Survey](https://arxiv.org/abs/1405.2227)
- [Texture Analysis for Liveness Detection](https://ieeexplore.ieee.org/document/6313548)
