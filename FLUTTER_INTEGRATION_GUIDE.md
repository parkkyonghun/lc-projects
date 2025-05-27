# 📱 Flutter Integration Guide for AI OCR Training

## 🎯 **Overview**

This guide shows you how to integrate your Flutter app with the AI OCR training system for camera-based training and real-time model improvement.

## 🚀 **Quick Start**

### **1. Add Dependencies to pubspec.yaml**
```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.1.0
  image_picker: ^1.0.4
  camera: ^0.10.5
  path_provider: ^2.1.1
  shared_preferences: ^2.2.2
  json_annotation: ^4.8.1
```

### **2. API Configuration**
```dart
class ApiConfig {
  static const String baseUrl = 'http://your-server:8000';
  static const String trainingEndpoint = '$baseUrl/training';
  static const String ocrEndpoint = '$baseUrl/ocr';
}
```

## 📸 **Camera-Based Training Implementation**

### **1. Training Service**
```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class TrainingService {
  static const String baseUrl = ApiConfig.baseUrl;
  
  // Start a new training session
  Future<String> startTrainingSession({
    required String sessionName,
    double targetAccuracy = 0.95,
    int maxSamples = 100,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/training/session/start'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'session_name': sessionName,
        'target_accuracy': targetAccuracy,
        'max_samples': maxSamples,
      }),
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['session_id'];
    } else {
      throw Exception('Failed to start training session');
    }
  }
  
  // Add training data from camera
  Future<Map<String, dynamic>> addCameraTrainingData({
    required String sessionId,
    required File imageFile,
    required Map<String, String> groundTruth,
    double? imageQuality,
  }) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/training/data/camera'),
    );
    
    // Add parameters
    request.fields['session_id'] = sessionId;
    request.fields['ground_truth'] = jsonEncode(groundTruth);
    if (imageQuality != null) {
      request.fields['image_quality'] = imageQuality.toString();
    }
    
    // Add image file
    request.files.add(
      await http.MultipartFile.fromPath('file', imageFile.path),
    );
    
    final response = await request.send();
    final responseBody = await response.stream.bytesToString();
    
    if (response.statusCode == 200) {
      return jsonDecode(responseBody);
    } else {
      throw Exception('Failed to add training data: $responseBody');
    }
  }
  
  // Get training progress
  Future<Map<String, dynamic>> getTrainingProgress(String sessionId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/training/progress/$sessionId'),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to get training progress');
    }
  }
  
  // Get model metrics
  Future<Map<String, dynamic>> getModelMetrics(String sessionId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/training/metrics/$sessionId'),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to get model metrics');
    }
  }
}
```

### **2. Camera Training Screen**
```dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class CameraTrainingScreen extends StatefulWidget {
  final String sessionId;
  
  const CameraTrainingScreen({Key? key, required this.sessionId}) : super(key: key);
  
  @override
  _CameraTrainingScreenState createState() => _CameraTrainingScreenState();
}

class _CameraTrainingScreenState extends State<CameraTrainingScreen> {
  final TrainingService _trainingService = TrainingService();
  final ImagePicker _picker = ImagePicker();
  
  File? _capturedImage;
  bool _isProcessing = false;
  Map<String, dynamic>? _trainingProgress;
  
  // Ground truth controllers
  final TextEditingController _nameKhController = TextEditingController();
  final TextEditingController _nameEnController = TextEditingController();
  final TextEditingController _idNumberController = TextEditingController();
  final TextEditingController _dobController = TextEditingController();
  final TextEditingController _genderController = TextEditingController();
  final TextEditingController _nationalityController = TextEditingController();
  
  @override
  void initState() {
    super.initState();
    _loadTrainingProgress();
  }
  
  Future<void> _loadTrainingProgress() async {
    try {
      final progress = await _trainingService.getTrainingProgress(widget.sessionId);
      setState(() {
        _trainingProgress = progress;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load progress: $e')),
      );
    }
  }
  
  Future<void> _captureImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
      );
      
      if (image != null) {
        setState(() {
          _capturedImage = File(image.path);
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to capture image: $e')),
      );
    }
  }
  
  Future<void> _submitTrainingData() async {
    if (_capturedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please capture an image first')),
      );
      return;
    }
    
    // Validate ground truth data
    if (_nameKhController.text.isEmpty || _idNumberController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please fill in required fields')),
      );
      return;
    }
    
    setState(() {
      _isProcessing = true;
    });
    
    try {
      final groundTruth = {
        'name_kh': _nameKhController.text,
        'name_en': _nameEnController.text,
        'id_number': _idNumberController.text,
        'date_of_birth': _dobController.text,
        'gender': _genderController.text,
        'nationality': _nationalityController.text,
      };
      
      final result = await _trainingService.addCameraTrainingData(
        sessionId: widget.sessionId,
        imageFile: _capturedImage!,
        groundTruth: groundTruth,
      );
      
      // Clear form and image
      _clearForm();
      
      // Reload progress
      await _loadTrainingProgress();
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Training data added successfully! ${result['training_triggered'] ? 'Training started.' : ''}'),
          backgroundColor: Colors.green,
        ),
      );
      
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to submit training data: $e')),
      );
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }
  
  void _clearForm() {
    setState(() {
      _capturedImage = null;
    });
    _nameKhController.clear();
    _nameEnController.clear();
    _idNumberController.clear();
    _dobController.clear();
    _genderController.clear();
    _nationalityController.clear();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Training - Camera'),
        backgroundColor: Colors.blue[600],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Training Progress Card
            if (_trainingProgress != null) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Training Progress',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: (_trainingProgress!['current_samples'] as int) / 
                               (_trainingProgress!['target_samples'] as int),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Samples: ${_trainingProgress!['current_samples']}/${_trainingProgress!['target_samples']}',
                      ),
                      Text(
                        'Accuracy: ${(_trainingProgress!['current_accuracy'] * 100).toStringAsFixed(1)}%',
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
            
            // Camera Section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Text(
                      'Capture ID Card',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 16),
                    
                    if (_capturedImage != null) ...[
                      Container(
                        height: 200,
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.grey),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Image.file(
                          _capturedImage!,
                          fit: BoxFit.contain,
                        ),
                      ),
                      const SizedBox(height: 16),
                    ],
                    
                    ElevatedButton.icon(
                      onPressed: _captureImage,
                      icon: const Icon(Icons.camera_alt),
                      label: Text(_capturedImage == null ? 'Capture Image' : 'Retake Image'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue[600],
                        foregroundColor: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Ground Truth Form
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Ground Truth Data',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 16),
                    
                    TextField(
                      controller: _nameKhController,
                      decoration: const InputDecoration(
                        labelText: 'Name (Khmer) *',
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    
                    TextField(
                      controller: _nameEnController,
                      decoration: const InputDecoration(
                        labelText: 'Name (English)',
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    
                    TextField(
                      controller: _idNumberController,
                      decoration: const InputDecoration(
                        labelText: 'ID Number *',
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    
                    TextField(
                      controller: _dobController,
                      decoration: const InputDecoration(
                        labelText: 'Date of Birth',
                        border: OutlineInputBorder(),
                        hintText: 'DD/MM/YYYY',
                      ),
                    ),
                    const SizedBox(height: 12),
                    
                    TextField(
                      controller: _genderController,
                      decoration: const InputDecoration(
                        labelText: 'Gender',
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    
                    TextField(
                      controller: _nationalityController,
                      decoration: const InputDecoration(
                        labelText: 'Nationality',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Submit Button
            ElevatedButton(
              onPressed: _isProcessing ? null : _submitTrainingData,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green[600],
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isProcessing
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text(
                      'Submit Training Data',
                      style: TextStyle(fontSize: 16),
                    ),
            ),
          ],
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _nameKhController.dispose();
    _nameEnController.dispose();
    _idNumberController.dispose();
    _dobController.dispose();
    _genderController.dispose();
    _nationalityController.dispose();
    super.dispose();
  }
}
```

## 📊 **Training Dashboard Widget**

```dart
class TrainingDashboardWidget extends StatefulWidget {
  final String sessionId;
  
  const TrainingDashboardWidget({Key? key, required this.sessionId}) : super(key: key);
  
  @override
  _TrainingDashboardWidgetState createState() => _TrainingDashboardWidgetState();
}

class _TrainingDashboardWidgetState extends State<TrainingDashboardWidget> {
  final TrainingService _trainingService = TrainingService();
  Map<String, dynamic>? _metrics;
  bool _isLoading = true;
  
  @override
  void initState() {
    super.initState();
    _loadMetrics();
    // Auto-refresh every 30 seconds
    Timer.periodic(const Duration(seconds: 30), (timer) {
      if (mounted) _loadMetrics();
    });
  }
  
  Future<void> _loadMetrics() async {
    try {
      final metrics = await _trainingService.getModelMetrics(widget.sessionId);
      setState(() {
        _metrics = metrics;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    
    if (_metrics == null) {
      return const Center(child: Text('Failed to load metrics'));
    }
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Model Performance',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 16),
            
            Row(
              children: [
                Expanded(
                  child: _MetricCard(
                    title: 'Accuracy',
                    value: '${(_metrics!['accuracy'] * 100).toStringAsFixed(1)}%',
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _MetricCard(
                    title: 'Precision',
                    value: '${(_metrics!['precision'] * 100).toStringAsFixed(1)}%',
                    color: Colors.green,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 8),
            
            Row(
              children: [
                Expanded(
                  child: _MetricCard(
                    title: 'Recall',
                    value: '${(_metrics!['recall'] * 100).toStringAsFixed(1)}%',
                    color: Colors.orange,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _MetricCard(
                    title: 'F1 Score',
                    value: '${(_metrics!['f1_score'] * 100).toStringAsFixed(1)}%',
                    color: Colors.purple,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String title;
  final String value;
  final Color color;
  
  const _MetricCard({
    required this.title,
    required this.value,
    required this.color,
  });
  
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          Text(
            title,
            style: TextStyle(
              fontSize: 12,
              color: color.withOpacity(0.8),
            ),
          ),
        ],
      ),
    );
  }
}
```

## 🎯 **Usage Example**

```dart
// Start a training session and navigate to camera training
Future<void> startTraining() async {
  final trainingService = TrainingService();
  
  try {
    final sessionId = await trainingService.startTrainingSession(
      sessionName: 'Cambodian ID Training ${DateTime.now().day}',
      targetAccuracy: 0.95,
      maxSamples: 50,
    );
    
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CameraTrainingScreen(sessionId: sessionId),
      ),
    );
  } catch (e) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Failed to start training: $e')),
    );
  }
}
```

## 🔧 **Configuration**

### **1. Permissions (android/app/src/main/AndroidManifest.xml)**
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

### **2. iOS Configuration (ios/Runner/Info.plist)**
```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to capture ID cards for training</string>
```

This Flutter integration provides a complete camera-based training solution that works seamlessly with your AI OCR training system!
