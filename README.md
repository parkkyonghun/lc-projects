# 🤖 AI OCR Training System for Cambodian ID Cards

## 🎯 **Overview**

A comprehensive AI-powered OCR system with advanced training capabilities specifically designed for Cambodian ID cards. This system provides both **Flutter integration** and **web-based training UI** options for continuous model improvement.

## ✨ **Key Features**

### **🎓 Advanced AI Training**
- **Camera-based training** with Flutter integration
- **Smart training orchestration** with active learning
- **Real-time model improvement** and performance monitoring
- **Quality-aware training** with automatic image assessment
- **Khmer script optimization** for Cambodian text

### **📱 Flutter Integration**
- **Camera capture API** for training data collection
- **Real-time progress tracking** with WebSocket support
- **Batch training capabilities** for efficient data processing
- **Model deployment endpoints** for production use
- **Comprehensive Flutter SDK** with example implementations

### **🌐 Smart Training UI**
- **Web-based dashboard** for training management
- **Live metrics visualization** with real-time updates
- **Training analytics** and performance insights
- **Session management** with progress tracking
- **AI-generated recommendations** for improvement

### **🖼️ Image Enhancement**
- **AI-powered enhancement** for ultra-low quality images
- **Extreme enhancement** for severely damaged documents
- **Quality assessment** and automatic optimization
- **Multi-level enhancement pipeline** for different quality levels

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd lc-projects

# Install dependencies
pip install -r requirements.txt

# Clean up project (optional)
python cleanup.py
```

### **2. Start the Server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **3. Access the System**
- **API Documentation**: http://localhost:8000/docs
- **Training Dashboard**: http://localhost:8000/ui/dashboard
- **OCR Endpoint**: http://localhost:8000/ocr/idcard
- **Health Check**: http://localhost:8000/health

## 📱 **Flutter Integration**

### **Option 1: Camera-Based Training**
Perfect for mobile apps that need to train AI models using camera captures.

```dart
// Start training session
final sessionId = await trainingService.startTrainingSession(
  sessionName: 'Mobile Training Session',
  targetAccuracy: 0.95,
  maxSamples: 100,
);

// Add camera training data
await trainingService.addCameraTrainingData(
  sessionId: sessionId,
  imageFile: capturedImage,
  groundTruth: correctFieldValues,
);
```

**📖 Complete Guide**: [FLUTTER_INTEGRATION_GUIDE.md](FLUTTER_INTEGRATION_GUIDE.md)

### **Option 2: Smart Training UI (Web)**
Perfect for desktop training, detailed analysis, and training management.

- **Access**: http://localhost:8000/ui/dashboard
- **Features**: Real-time monitoring, analytics, session management
- **Best for**: Detailed training analysis, bulk data processing

## 🎯 **Training Workflow**

### **1. Start Training Session**
```bash
POST /training/session/start
{
  "session_name": "Cambodian ID Training v1",
  "target_accuracy": 0.95,
  "max_samples": 100
}
```

### **2. Add Training Data**
```bash
# Camera-based (Flutter)
POST /training/data/camera

# Batch upload
POST /training/data/batch
```

### **3. Monitor Progress**
```bash
GET /training/progress/{session_id}
GET /training/metrics/{session_id}
```

### **4. Deploy Model**
```bash
POST /training/model/deploy/{session_id}
```

## 📊 **API Endpoints**

### **OCR Processing**
- `POST /ocr/idcard` - Process Cambodian ID card with AI enhancement

### **Training Management**
- `POST /training/session/start` - Start new training session
- `POST /training/data/camera` - Add camera training data
- `POST /training/data/batch` - Batch training data upload
- `GET /training/progress/{session_id}` - Get training progress
- `GET /training/metrics/{session_id}` - Get model metrics
- `POST /training/model/deploy/{session_id}` - Deploy trained model

### **Smart Training UI**
- `GET /ui/dashboard` - Web training dashboard
- `GET /ui/api/dashboard/stats` - Dashboard statistics
- `WebSocket /ui/ws/training/{session_id}` - Real-time updates

## 🏗️ **Project Structure**

```
lc-projects/
├── 🎯 Core System
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Dependencies
│   └── cleanup.py                  # Project cleanup script
│
├── 🔧 Core Modules
│   ├── core/                       # Core functionality
│   │   ├── training_manager.py     # Training session management
│   │   └── model_evaluator.py     # Performance evaluation
│   ├── api/                        # API endpoints
│   │   ├── training_endpoints.py   # Flutter integration APIs
│   │   └── smart_training_ui.py    # Web dashboard APIs
│   └── training/                   # Training modules
│
├── 🎨 MVC Architecture
│   ├── models/                     # Data models
│   ├── views/                      # API routes
│   ├── controllers/                # Business logic
│   └── schemas/                    # Pydantic schemas
│
├── 🎓 AI Training System
│   ├── ai_training_system.py       # Base training system
│   ├── advanced_ai_training.py     # Advanced features
│   ├── smart_training_orchestrator.py # Intelligent orchestration
│   └── khmer_model_trainer.py      # Khmer-specific training
│
└── 📚 Documentation
    ├── PROJECT_STRUCTURE.md
    ├── FLUTTER_INTEGRATION_GUIDE.md
    └── README.md
```

**📖 Detailed Structure**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 🎓 **Training for 95%+ Accuracy**

### **Recommended Approach**
1. **Start with 20-30 clear, high-quality ID card images**
2. **Use the Flutter camera integration for real-time training**
3. **Gradually add challenging samples (poor lighting, angles, quality)**
4. **Monitor field-specific accuracy and focus on weak areas**
5. **Use active learning to select the most valuable training samples**

### **Training Tips**
- **Quality over quantity**: 50 high-quality samples > 200 poor samples
- **Diverse conditions**: Different lighting, angles, image qualities
- **Field balance**: Ensure all fields (name, ID, date, etc.) are well represented
- **Continuous monitoring**: Use the dashboard to track progress
- **Incremental improvement**: Deploy models incrementally as they improve

## 📈 **Performance Monitoring**

### **Real-time Metrics**
- **Accuracy tracking** per field and overall
- **Processing time** optimization
- **Quality improvement** analysis
- **Training progress** visualization

### **Analytics Dashboard**
- **Session overview** with success rates
- **Performance trends** over time
- **Improvement recommendations** from AI
- **Model comparison** and selection

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Optional: Create .env file
DATABASE_URL=sqlite:///./training_data/training.db
MODEL_PATH=./khmer_models/
TRAINING_DATA_PATH=./training_data/
```

### **Training Configuration**
```python
# Modify ai_enhancement_config.py for custom settings
AI_TRAINING_CONFIG = {
    "target_accuracy": 0.95,
    "max_training_samples": 100,
    "quality_threshold": 0.3,
    "active_learning": True
}
```

## 🧪 **Testing**

```bash
# Run OCR tests
python -m pytest tests/

# Test training system
python training_data_collector.py quick

# Test image enhancement
python test_image_enhancement.py
```

## 🚀 **Deployment**

### **Production Setup**
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **Docker Deployment**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🎯 **Next Steps**

1. **Choose your training approach**:
   - **Flutter Integration**: For mobile camera-based training
   - **Web Dashboard**: For desktop training and analysis

2. **Start training**:
   - Collect 20-30 high-quality ID card images
   - Use the training APIs or dashboard
   - Monitor progress and accuracy

3. **Deploy your model**:
   - Achieve 95%+ accuracy
   - Deploy for production use
   - Continue monitoring and improving

**Happy Training! 🚀**

## Setup Instructions (Legacy)
   cd lc-projects
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

The main entry point is `main.py`. To start the FastAPI server, run:

```bash
uvicorn main:app --reload
```

- The API will be available at: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

## Project Structure

- `main.py` - Application entry point
- `controllers/` - Route controllers
- `models/` - Database models
- `schemas/` - Pydantic schemas
- `views/` - (If used) View logic
- `tests/` - Test cases

## Environment Variables
If your project uses environment variables, create a `.env` file in the root directory and add your variables there.

---
If you have any questions or need additional setup, let me know!
