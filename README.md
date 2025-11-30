# 🔐 Facial Recognition Service

A modern FastAPI-based facial recognition service with **auto-enrollment**, liveness detection, and FAISS indexing for fast face matching. Features a beautiful web interface for easy testing.

## ✨ Features

- **🎯 Auto-Enrollment** - New users are automatically enrolled on first verification
- **👤 Face Detection** - Robust face detection with OpenCV Haar Cascades
- **✅ Face Verification** - 1:1 matching to verify user identity  
- **🔍 Face Identification** - 1:N matching to identify unknown faces
- **🛡️ Liveness Detection** - Passive liveness checks to prevent spoofing
- **⚡ Fast Search** - FAISS-based similarity search for instant matching
- **🎨 Modern Web UI** - Beautiful, responsive interface for testing
- **� FFile Upload Support** - Direct image upload (no manual base64 encoding)
- **🌐 RESTful API** - Clean, well-documented API endpoints
- **🔒 Strict Validation** - Only accepts images with detectable faces

## � Quicek Start

### Prerequisites

**macOS:**
```bash
brew install libjpeg zlib
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3-dev libjpeg-dev zlib1g-dev
```

### Installation

1. **Create virtual environment:**
```bash
cd facial-recognition-service
python3 -m venv faiss_env
source faiss_env/bin/activate
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Run the service:**
```bash
uvicorn app.main:app --reload
```

4. **Open the web interface:**
```bash
open test_upload.html
```

## 🎨 Web Interface

Beautiful dark-mode interface with:
- Animated gradient background
- Drag & drop image upload
- Real-time image preview
- Rich result display with badges and stats
- Responsive design

## 📡 API Endpoints

### Verify Face (Auto-Enroll)
```http
POST /recognition/verify
Content-Type: multipart/form-data

user_id: john_doe
file: [image file]
```

### Enroll Face
```http
POST /recognition/enroll
Content-Type: multipart/form-data

user_id: john_doe
file: [image file]
```

### Identify Face
```http
POST /recognition/identify
Content-Type: multipart/form-data

file: [image file]
```

## 📚 API Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## 🧪 Testing with cURL

```bash
curl -X POST "http://localhost:8000/recognition/verify" \
  -F "user_id=john_doe" \
  -F "file=@path/to/face.jpg"
```

## ⚙️ Configuration

Create `.env` file:

```env
DATABASE_URL=sqlite:///./faces.db
FAISS_INDEX_PATH=./faiss_index.bin
SIMILARITY_THRESHOLD=0.6
IMAGE_MAX_SIZE=2048
```

## 🏗️ Architecture

```
facial-recognition-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── database.py          # Database setup
│   ├── models.py            # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   ├── face_service.py      # Face recognition logic
│   ├── faiss_index.py       # FAISS indexing
│   ├── routers/
│   │   └── recognition.py   # API endpoints
│   └── utils/
│       ├── image_utils.py   # Image processing
│       └── liveness.py      # Liveness detection
├── test_upload.html         # Web interface
└── requirements.txt
```

## 🔒 Security Considerations

This is a **demonstration** implementation. For production:

1. **Add Authentication** - API keys, OAuth2/JWT
2. **Use Real Face Recognition** - DeepFace, FaceNet, ArcFace, InsightFace
3. **Enhanced Liveness** - Active detection, 3D depth sensing
4. **Data Protection** - Encrypt biometric data, HTTPS, GDPR compliance
5. **Production Infrastructure** - PostgreSQL, Redis, rate limiting, monitoring
6. **Security Hardening** - Input validation, CORS, DDoS protection

## 📊 Response Examples

**Successful Auto-Enrollment:**
```json
{
  "verified": true,
  "confidence": 1.0,
  "liveness_passed": true,
  "auto_enrolled": true,
  "user_exists": false,
  "message": "New user enrolled successfully"
}
```

**Successful Verification:**
```json
{
  "verified": true,
  "confidence": 0.95,
  "liveness_passed": true,
  "auto_enrolled": false,
  "user_exists": true,
  "message": "Face verified successfully"
}
```

**No Face Detected:**
```json
{
  "detail": "No face detected in the image. Please upload a clear photo showing your face."
}
```

## ⚠️ Disclaimer

This is a demonstration project. The placeholder face recognition is not suitable for production. Integrate a real face recognition model for actual use.

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework
- **FAISS** - Efficient similarity search
- **OpenCV** - Computer vision library
- **SQLAlchemy** - Database ORM

---

**Built with ❤️ using FastAPI**
