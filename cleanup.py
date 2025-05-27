#!/usr/bin/env python3
"""
Project Cleanup Script

This script cleans up temporary files, cache files, and organizes the project structure.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_pycache():
    """Remove all __pycache__ directories."""
    print("🧹 Cleaning up __pycache__ directories...")
    
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"  Removing: {pycache_path}")
            shutil.rmtree(pycache_path)
    
    print("✅ __pycache__ cleanup completed")

def cleanup_temp_files():
    """Remove temporary files and test artifacts."""
    print("🧹 Cleaning up temporary files...")
    
    temp_patterns = [
        "*.pyc",
        "*.pyo",
        "*.tmp",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "*.log"
    ]
    
    for pattern in temp_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            print(f"  Removing: {file}")
            os.remove(file)
    
    print("✅ Temporary files cleanup completed")

def organize_training_data():
    """Organize training data directory structure."""
    print("📁 Organizing training data structure...")
    
    training_dir = Path("training_data")
    
    # Create organized subdirectories
    subdirs = [
        "sessions",
        "raw_images", 
        "processed_images",
        "annotations",
        "models",
        "metrics",
        "exports"
    ]
    
    for subdir in subdirs:
        (training_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("✅ Training data organization completed")

def create_gitignore():
    """Create or update .gitignore file."""
    print("📝 Creating/updating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Training Data (keep structure, ignore content)
training_data/raw_images/*.jpg
training_data/raw_images/*.png
training_data/processed_images/*.jpg
training_data/processed_images/*.png
training_data/sessions/*/
training_data/*.db

# Test Results
test_results/
extreme_test_results/
extreme_ocr_results/

# Temporary Files
*.tmp
*.log
*~

# Model Files (large files)
*.pkl
*.h5
*.pth
*.onnx

# Environment Variables
.env
.env.local
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created/updated")

def display_project_structure():
    """Display the cleaned project structure."""
    print("\n🏗️  Clean Project Structure:")
    print("=" * 50)
    
    structure = """
lc-projects/
├── 🎯 Core System
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Dependencies
│   └── cleanup.py                  # This cleanup script
│
├── 🔧 Core Modules
│   ├── core/                       # Core functionality
│   ├── api/                        # API endpoints
│   └── training/                   # Training modules
│
├── 🎨 MVC Architecture
│   ├── models/                     # Data models
│   ├── views/                      # API routes
│   ├── controllers/                # Business logic
│   └── schemas/                    # Pydantic schemas
│
├── 🎓 AI Training System
│   ├── ai_training_system.py       # Base training
│   ├── advanced_ai_training.py     # Advanced features
│   ├── smart_training_orchestrator.py
│   └── khmer_model_trainer.py      # Khmer-specific
│
├── 🖼️ Image Enhancement
│   ├── ai_image_enhancement.py
│   ├── extreme_enhancement.py
│   └── image_enhancement_utils.py
│
├── 📊 Data & Storage
│   ├── training_data/              # Organized training data
│   └── khmer_models/               # Trained models
│
├── 🌐 Web Interface
│   └── templates/                  # HTML templates
│
└── 📚 Documentation
    ├── PROJECT_STRUCTURE.md
    ├── FLUTTER_INTEGRATION_GUIDE.md
    └── README.md
"""
    
    print(structure)

def main():
    """Main cleanup function."""
    print("🚀 Starting Project Cleanup...")
    print("=" * 50)
    
    # Perform cleanup tasks
    cleanup_pycache()
    cleanup_temp_files()
    organize_training_data()
    create_gitignore()
    
    # Display results
    display_project_structure()
    
    print("\n✨ Project cleanup completed successfully!")
    print("\n🎯 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start the server: uvicorn main:app --reload")
    print("3. Access training dashboard: http://localhost:8000/ui/dashboard")
    print("4. Check API docs: http://localhost:8000/docs")
    print("5. Integrate with Flutter using the guide: FLUTTER_INTEGRATION_GUIDE.md")

if __name__ == "__main__":
    main()
