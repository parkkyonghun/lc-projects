#!/usr/bin/env python3
"""
Server Startup Script

This script starts the AI OCR Training System server with proper error handling.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    print("🏗️  Checking project structure...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "templates/training_dashboard.html",
        "templates/training_guide.html"
    ]
    
    required_dirs = [
        "api",
        "core", 
        "schemas",
        "views",
        "controllers",
        "models",
        "templates"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    for dir in required_dirs:
        if not Path(dir).exists():
            missing_dirs.append(dir)
    
    if missing_files or missing_dirs:
        print("❌ Missing project components:")
        for file in missing_files:
            print(f"   📄 {file}")
        for dir in missing_dirs:
            print(f"   📁 {dir}/")
        return False
    
    print("✅ Project structure is correct")
    return True

def test_imports():
    """Test if main modules can be imported."""
    print("📦 Testing module imports...")
    
    try:
        # Test main app import
        import main
        print("✅ Main app imports successfully")
        
        # Test if FastAPI app is created
        if hasattr(main, 'app'):
            print("✅ FastAPI app created successfully")
        else:
            print("⚠️  FastAPI app not found in main module")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def start_server():
    """Start the FastAPI server."""
    print("🚀 Starting AI OCR Training System server...")
    print("=" * 60)
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        print("📡 Server starting on: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🌐 Training Dashboard: http://localhost:8000/ui/dashboard")
        print("🎓 Training Guide: http://localhost:8000/ui/training-guide")
        print("💚 Health Check: http://localhost:8000/health")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        print()
        
        # Run the server
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("🤖 AI OCR Training System - Server Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    # Check project structure
    if not check_project_structure():
        print("\n💡 To fix project structure:")
        print("   python cleanup.py")
        return False
    
    # Test imports
    if not test_imports():
        print("\n💡 Check for import errors in your code")
        return False
    
    print("✅ All checks passed!")
    print()
    
    # Start server
    return start_server()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
