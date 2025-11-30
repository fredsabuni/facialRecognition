import numpy as np
import cv2
from typing import Optional, Tuple

def check_liveness(image: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if the face in the image is from a live person using passive detection.
    Combines multiple heuristics to detect spoofing attempts.
    
    Args:
        image: Input image as numpy array (BGR format)
        threshold: Confidence threshold for liveness (0-1)
    
    Returns:
        True if face appears to be live, False otherwise
    """
    if image is None or image.size == 0:
        return False
    
    # Calculate liveness score using multiple checks
    liveness_score = calculate_liveness_score(image)
    
    return liveness_score >= threshold


def calculate_liveness_score(image: np.ndarray) -> float:
    """
    Calculate overall liveness score by combining multiple detection methods.
    
    Returns:
        Score between 0 (likely spoof) and 1 (likely live)
    """
    scores = []
    
    # 1. Texture analysis - real faces have more texture variation
    texture_score = analyze_texture(image)
    scores.append(texture_score)
    
    # 2. Color diversity - printed photos have less color variation
    color_score = analyze_color_diversity(image)
    scores.append(color_score)
    
    # 3. Frequency analysis - screens/prints have different frequency patterns
    frequency_score = analyze_frequency(image)
    scores.append(frequency_score)
    
    # 4. Sharpness check - blurry images might be from screens
    sharpness_score = analyze_sharpness(image)
    scores.append(sharpness_score)
    
    # Weighted average of all scores
    weights = [0.3, 0.25, 0.25, 0.2]
    overall_score = sum(s * w for s, w in zip(scores, weights))
    
    return overall_score


def analyze_texture(image: np.ndarray) -> float:
    """
    Analyze texture patterns using Local Binary Patterns (LBP).
    Real faces have richer texture than printed photos.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate LBP-like texture measure
    # Compare each pixel with its neighbors
    rows, cols = gray.shape
    if rows < 3 or cols < 3:
        return 0.5
    
    # Simple texture variance calculation
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)
    
    # Normalize score (higher variance = more texture = more likely live)
    # Typical range: 100-10000 for live faces, 10-100 for prints
    score = min(texture_variance / 5000.0, 1.0)
    
    return score


def analyze_color_diversity(image: np.ndarray) -> float:
    """
    Analyze color distribution. Live faces have more color variation
    than printed photos or screens.
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate color histogram diversity
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    # Normalize histograms
    h_hist = h_hist.flatten() / h_hist.sum()
    s_hist = s_hist.flatten() / s_hist.sum()
    
    # Calculate entropy (higher entropy = more diversity)
    h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-10))
    s_entropy = -np.sum(s_hist * np.log2(s_hist + 1e-10))
    
    # Normalize scores
    h_score = min(h_entropy / 7.0, 1.0)  # Max entropy ~7 for 180 bins
    s_score = min(s_entropy / 8.0, 1.0)  # Max entropy ~8 for 256 bins
    
    return (h_score + s_score) / 2.0


def analyze_frequency(image: np.ndarray) -> float:
    """
    Analyze frequency domain. Screens and prints have characteristic
    frequency patterns (moiré patterns, pixel grids).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Analyze high-frequency content
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask for high frequencies (outer region)
    mask = np.ones((rows, cols), dtype=np.uint8)
    r = min(rows, cols) // 4
    cv2.circle(mask, (ccol, crow), r, 0, -1)
    
    # Calculate high-frequency energy
    high_freq_energy = np.sum(magnitude * mask)
    total_energy = np.sum(magnitude)
    
    # Live faces typically have more high-frequency content
    ratio = high_freq_energy / (total_energy + 1e-10)
    
    # Normalize (typical range 0.3-0.7 for live, 0.1-0.3 for spoof)
    score = min(max((ratio - 0.1) / 0.4, 0.0), 1.0)
    
    return score


def analyze_sharpness(image: np.ndarray) -> float:
    """
    Analyze image sharpness. Blurry images might indicate
    a photo of a photo or screen capture.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (measure of sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize score (typical range: 100-1000 for sharp, <100 for blurry)
    score = min(laplacian_var / 500.0, 1.0)
    
    return score


def detect_spoofing(image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect spoofing attempts (photos, videos, masks).
    
    Returns:
        Tuple of (is_spoof, confidence_score)
        - is_spoof: True if spoofing detected
        - confidence_score: Confidence of the detection (0-1)
    """
    liveness_score = calculate_liveness_score(image)
    
    # Invert score for spoofing (low liveness = high spoof probability)
    spoof_score = 1.0 - liveness_score
    is_spoof = spoof_score > 0.5
    
    return is_spoof, spoof_score
