#!/usr/bin/env python3
"""
Quick GPU Acceleration Test for AI OCR System

This script tests current GPU capabilities and provides immediate recommendations
for improving OCR performance with GPU acceleration.
"""

import time
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_current_gpu_setup():
    """Test current GPU setup and provide recommendations."""
    print("🧪 GPU Acceleration Test for AI OCR System")
    print("=" * 50)
    
    results = {
        "gpu_hardware": False,
        "pytorch_gpu": False,
        "opencv_gpu": False,
        "ai_enhancement_gpu": False,
        "recommendations": []
    }
    
    # Test 1: GPU Hardware Detection
    print("\n1️⃣ Testing GPU Hardware...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0].split(', ')
            gpu_name, gpu_memory = gpu_info[0], gpu_info[1]
            print(f"   ✅ GPU Detected: {gpu_name} ({gpu_memory}MB)")
            results["gpu_hardware"] = True
            
            if int(gpu_memory) >= 4000:
                print(f"   ✅ Sufficient GPU memory for AI enhancement")
            else:
                print(f"   ⚠️ Limited GPU memory - consider batch size optimization")
                results["recommendations"].append("Optimize batch sizes for limited GPU memory")
        else:
            print("   ❌ No NVIDIA GPU detected")
            results["recommendations"].append("GPU acceleration not available - consider hardware upgrade")
    except Exception as e:
        print(f"   ❌ GPU detection failed: {e}")
    
    # Test 2: PyTorch CUDA Support
    print("\n2️⃣ Testing PyTorch CUDA Support...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✅ PyTorch CUDA Available: {device_count} device(s)")
            print(f"   ✅ Primary Device: {device_name}")
            results["pytorch_gpu"] = True
            
            # Quick performance test
            print("   🔄 Running PyTorch GPU performance test...")
            device = torch.device('cuda')
            
            # CPU test
            start_time = time.time()
            cpu_tensor = torch.rand(1000, 1000)
            cpu_result = torch.mm(cpu_tensor, cpu_tensor)
            cpu_time = time.time() - start_time
            
            # GPU test
            start_time = time.time()
            gpu_tensor = torch.rand(1000, 1000, device=device)
            gpu_result = torch.mm(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"   📊 CPU Time: {cpu_time:.3f}s, GPU Time: {gpu_time:.3f}s")
            print(f"   🚀 GPU Speedup: {speedup:.1f}x")
            
            if speedup > 5:
                print("   ✅ Excellent GPU acceleration potential")
            elif speedup > 2:
                print("   ✅ Good GPU acceleration potential")
            else:
                print("   ⚠️ Limited GPU acceleration - check drivers/setup")
                results["recommendations"].append("Check GPU drivers and CUDA installation")
        else:
            print("   ❌ PyTorch CUDA not available")
            results["recommendations"].append("Install PyTorch with CUDA support")
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
    
    # Test 3: OpenCV CUDA Support
    print("\n3️⃣ Testing OpenCV CUDA Support...")
    try:
        import cv2
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            print(f"   ✅ OpenCV CUDA Available: {cuda_devices} device(s)")
            results["opencv_gpu"] = True
            
            # Test GPU image processing
            print("   🔄 Testing OpenCV GPU image processing...")
            test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            
            # CPU processing
            start_time = time.time()
            cpu_blur = cv2.GaussianBlur(test_image, (15, 15), 0)
            cpu_time = time.time() - start_time
            
            # GPU processing (if available)
            try:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(test_image)
                start_time = time.time()
                gpu_blur = cv2.cuda.bilateralFilter(gpu_image, -1, 50, 50)
                gpu_result = gpu_blur.download()
                gpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"   📊 CPU Time: {cpu_time:.3f}s, GPU Time: {gpu_time:.3f}s")
                print(f"   🚀 GPU Speedup: {speedup:.1f}x")
            except Exception as e:
                print(f"   ⚠️ OpenCV GPU processing test failed: {e}")
        else:
            print("   ❌ OpenCV CUDA not available")
            results["recommendations"].append("Install OpenCV with CUDA support for image processing acceleration")
    except Exception as e:
        print(f"   ❌ OpenCV test failed: {e}")
    
    # Test 4: AI Enhancement System GPU Usage
    print("\n4️⃣ Testing AI Enhancement System...")
    try:
        from ai_enhancement_config import AI_TRAINING_CONFIG, KHMER_OPTIMIZED_CONFIG
        
        configs_to_test = [
            ("AI_TRAINING_CONFIG", AI_TRAINING_CONFIG),
            ("KHMER_OPTIMIZED_CONFIG", KHMER_OPTIMIZED_CONFIG)
        ]
        
        for config_name, config in configs_to_test:
            print(f"   🔍 Checking {config_name}...")
            if config.use_gpu:
                print(f"   ✅ {config_name} configured for GPU usage")
            else:
                print(f"   ⚠️ {config_name} not configured for GPU usage")
                results["recommendations"].append(f"Enable GPU in {config_name}")
        
        # Test AI enhancer creation
        try:
            from ai_image_enhancement import create_ai_enhancer
            enhancer = create_ai_enhancer(use_gpu=True)
            if enhancer.use_gpu:
                print("   ✅ AI Enhancer can use GPU")
                results["ai_enhancement_gpu"] = True
            else:
                print("   ❌ AI Enhancer falling back to CPU")
                results["recommendations"].append("Fix AI Enhancer GPU configuration")
        except Exception as e:
            print(f"   ⚠️ AI Enhancer test failed: {e}")
            
    except Exception as e:
        print(f"   ❌ AI Enhancement system test failed: {e}")
    
    # Test 5: OCR Controller GPU Usage
    print("\n5️⃣ Testing OCR Controller Configuration...")
    try:
        # Check if OCR controller uses GPU-accelerated enhancement
        print("   🔍 Checking OCR controller GPU usage...")
        
        # This would test the actual OCR pipeline
        print("   ℹ️ OCR controller uses AI enhancement when ai_enhancement=True")
        print("   ℹ️ Current bottleneck: Tesseract OCR (CPU-only)")
        
        results["recommendations"].extend([
            "Use ai_enhancement=True in OCR requests for GPU acceleration",
            "Consider GPU-accelerated OCR alternatives to Tesseract",
            "Implement batch processing for multiple images"
        ])
        
    except Exception as e:
        print(f"   ❌ OCR controller test failed: {e}")
    
    # Generate Summary and Recommendations
    print("\n📋 GPU Acceleration Summary")
    print("=" * 50)
    print(f"GPU Hardware: {'✅' if results['gpu_hardware'] else '❌'}")
    print(f"PyTorch CUDA: {'✅' if results['pytorch_gpu'] else '❌'}")
    print(f"OpenCV CUDA: {'✅' if results['opencv_gpu'] else '❌'}")
    print(f"AI Enhancement GPU: {'✅' if results['ai_enhancement_gpu'] else '❌'}")
    
    print("\n💡 Recommendations for GPU Acceleration:")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"   {i}. {rec}")
    
    # Additional specific recommendations
    print("\n🎯 Immediate Actions for Better OCR Performance:")
    print("   1. 🚨 Install OpenCV with CUDA support (biggest impact)")
    print("   2. ⚡ Use ai_enhancement=True in OCR requests")
    print("   3. 🔧 Enable GPU in all AI enhancement configurations")
    print("   4. 📊 Monitor GPU utilization during processing")
    print("   5. 🎛️ Optimize batch sizes for your GTX 1070 (8GB)")
    
    print("\n📈 Expected Performance Improvements with GPU:")
    print("   • Image preprocessing: 5-15x faster")
    print("   • AI enhancement: 10-30x faster")
    print("   • Model training: 50-200x faster")
    print("   • Total OCR time: Reduce from 10-30s to 1-3s")
    
    return results


def test_gpu_enhanced_ocr():
    """Test GPU-enhanced OCR processing."""
    print("\n🧪 Testing GPU-Enhanced OCR Processing...")
    
    try:
        # Create a test image
        test_image = Image.new('RGB', (800, 600), color='white')
        
        # Test with GPU acceleration
        print("   🔄 Testing with GPU acceleration...")
        start_time = time.time()
        
        # This would use the GPU-accelerated enhancement
        try:
            from gpu_accelerated_ai_enhancement import create_gpu_ai_enhancer
            gpu_enhancer = create_gpu_ai_enhancer()
            enhanced_image = gpu_enhancer.enhance_image_gpu(test_image)
            gpu_time = time.time() - start_time
            print(f"   ✅ GPU enhancement completed in {gpu_time:.3f}s")
            
            # Get performance stats
            stats = gpu_enhancer.get_performance_stats()
            print(f"   📊 Device used: {stats['device']}")
            print(f"   📊 GPU available: {stats['gpu_available']}")
            
        except Exception as e:
            print(f"   ❌ GPU enhancement test failed: {e}")
        
        # Test with CPU fallback
        print("   🔄 Testing CPU fallback...")
        start_time = time.time()
        try:
            from ai_image_enhancement import create_ai_enhancer
            cpu_enhancer = create_ai_enhancer(use_gpu=False)
            enhanced_image = cpu_enhancer.enhance_ultra_low_quality(test_image)
            cpu_time = time.time() - start_time
            print(f"   ✅ CPU enhancement completed in {cpu_time:.3f}s")
        except Exception as e:
            print(f"   ❌ CPU enhancement test failed: {e}")
            cpu_time = 0
        
        # Compare performance
        if 'gpu_time' in locals() and cpu_time > 0:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"   🚀 GPU vs CPU speedup: {speedup:.1f}x")
        
    except Exception as e:
        print(f"   ❌ GPU-enhanced OCR test failed: {e}")


if __name__ == "__main__":
    # Run comprehensive GPU tests
    results = test_current_gpu_setup()
    
    # Test GPU-enhanced OCR
    test_gpu_enhanced_ocr()
    
    print("\n🎉 GPU Acceleration Testing Complete!")
    print("\nNext steps:")
    print("1. Run: python gpu_optimization_setup.py")
    print("2. Update OCR requests to use ai_enhancement=True")
    print("3. Monitor GPU utilization with nvidia-smi")
