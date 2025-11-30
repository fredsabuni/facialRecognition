#!/usr/bin/env python3
"""Test script to make API requests"""

import base64
import requests
import sys
import json

def test_verify_endpoint(image_path, user_id="test_user_123", base_url="http://localhost:8000"):
    """Test the /verify endpoint"""
    
    print(f"Testing /verify endpoint...")
    print(f"Image: {image_path}")
    print(f"User ID: {user_id}")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"✓ Image encoded. Base64 length: {len(base64_string)}")
    print(f"  First 50 chars: {base64_string[:50]}")
    
    # Prepare request
    url = f"{base_url}/recognition/verify"
    payload = {
        "user_id": user_id,
        "image": base64_string
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\n📤 Sending POST request to: {url}")
    print(f"  Payload keys: {list(payload.keys())}")
    print(f"  user_id: {payload['user_id']}")
    print(f"  image length: {len(payload['image'])}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"\n📥 Response received:")
        print(f"  Status Code: {response.status_code}")
        print(f"  Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"\n  Response Body:")
            print(json.dumps(response_json, indent=2))
            
            if response.status_code == 200:
                print("\n✅ Request successful!")
                if response_json.get('auto_enrolled'):
                    print("  → User was auto-enrolled")
                if response_json.get('verified'):
                    print(f"  → Face verified with confidence: {response_json.get('confidence')}")
                if response_json.get('liveness_passed'):
                    print("  → Liveness check passed")
            else:
                print(f"\n❌ Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"  Response Body (raw): {response.text}")
            print(f"  Error parsing JSON: {e}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to server")
        print("   Make sure the server is running on", base_url)
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_enroll_endpoint(image_path, user_id="test_user_456", base_url="http://localhost:8000"):
    """Test the /enroll endpoint"""
    
    print(f"\nTesting /enroll endpoint...")
    print(f"Image: {image_path}")
    print(f"User ID: {user_id}")
    print("-" * 50)
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare request
    url = f"{base_url}/recognition/enroll"
    payload = {
        "user_id": user_id,
        "image": base64_string
    }
    
    print(f"📤 Sending POST request to: {url}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"📥 Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Enrollment successful!")
        else:
            print(f"❌ Enrollment failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_api.py <image_file> [user_id] [base_url]")
        print("\nExample:")
        print("  python test_api.py sample_test.png")
        print("  python test_api.py sample_test.png my_user_123")
        print("  python test_api.py sample_test.png my_user_123 http://localhost:8000")
        sys.exit(1)
    
    image_path = sys.argv[1]
    user_id = sys.argv[2] if len(sys.argv) > 2 else "test_user_123"
    base_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    
    # Test verify endpoint (with auto-enroll)
    test_verify_endpoint(image_path, user_id, base_url)
    
    print("\n" + "=" * 50)
    print("Test complete!")
