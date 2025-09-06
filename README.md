# 🎯 YOLOv8 Custom Object Detection System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/yolov8-detection-system/blob/main/YOLOv8_Detection_System.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)

A complete end-to-end object detection system built with YOLOv8 in Jupyter Notebook format, featuring automated dataset processing, model training, real-time inference, and an interactive web dashboard - all runnable in Google Colab!

## 🚀 Features

### 📓 **Complete Jupyter Notebook Pipeline**
- **Single notebook execution** - run everything in one place
- **Google Colab optimized** with GPU acceleration
- **Interactive cells** for step-by-step execution
- **Real-time progress tracking** with visual outputs

### 🔧 **End-to-End Workflow**
- **Automated COCO to YOLO conversion** with proper dataset splitting
- **YOLOv8 model training** with live progress monitoring
- **Real-time object detection** with confidence thresholding
- **Interactive Gradio dashboard** embedded in notebook

### 📊 **Built-in Visualization**
- Training progress plots and metrics
- Real-time FPS tracking and performance graphs
- Interactive detection interface
- Comprehensive performance dashboards

### 💾 **Google Drive Integration**
- Automatic mounting and data access
- Model and results auto-saving
- Persistent storage between sessions

## 📁 Notebook Structure

```
YOLOv8_Detection_System.ipynb
├── 🔧 Setup & Dependencies          # Install required packages
├── 📂 Google Drive Integration      # Mount drive and setup paths
├── 🔄 Dataset Conversion            # COCO to YOLO format conversion
├── 🎯 Model Training               # YOLOv8 training with progress tracking
├── 📊 Performance Evaluation       # Metrics calculation and visualization
├── 🎛️ Interactive Dashboard        # Gradio web interface
├── 🔍 Real-time Detection         # Live inference testing
└── 💾 Save & Export               # Save models and results to Drive
```

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended)**

1. **Click the "Open in Colab" badge above**
2. **Connect to GPU runtime:**
   - Runtime → Change runtime type → GPU → Save
3. **Run all cells sequentially** or use Runtime → Run all
4. **Access your trained model and dashboard**

### **Option 2: Local Jupyter**

1. **Clone and setup:**
```bash
git clone https://github.com/yourusername/yolov8-detection-system.git
cd yolov8-detection-system
pip install -r requirements.txt
jupyter notebook YOLOv8_Detection_System.ipynb
```

2. **Run cells sequentially**

## 📋 Prerequisites

### **For Google Colab (Recommended)**
- Google account
- Google Drive with ~2GB free space
- Web browser

### **For Local Execution**
- Python 3.8+
- Jupyter Notebook/Lab
- CUDA-capable GPU (recommended)
- 4GB+ RAM

## 🔧 Notebook Sections

### **1. 🔧 Setup & Dependencies**
```python
# Installs all required packages
!pip install ultralytics torch torchvision opencv-python pillow matplotlib seaborn plotly --quiet
!pip install roboflow supervision gradio --quiet
```

### **2. 📂 Google Drive Integration**
```python
# Mounts Google Drive for data persistence
from google.colab import drive
drive.mount('/content/drive')
```

### **3. 🔄 Dataset Conversion**
- Converts COCO format to YOLO format
- Automated train/val/test splitting
- Creates dataset.yaml configuration

### **4. 🎯 Model Training**
- YOLOv8 Nano training (optimized for Colab)
- Real-time training progress visualization
- Automatic best model saving

### **5. 📊 Performance Evaluation**
- Comprehensive metrics calculation
- Interactive performance dashboard
- Visual training results analysis

### **6. 🎛️ Interactive Dashboard**
- Gradio web interface for real-time detection
- Multi-tab interface with different features
- Live performance monitoring

### **7. 💾 Save & Export**
- Automatic saving to Google Drive
- Model export for production use
- Results archiving

## 📊 Expected Performance

### **Training Results** (30 epochs on sample dataset)
- **mAP@50**: ~0.85 (85%)
- **mAP@50-95**: ~0.62 (62%)
- **Training Time**: ~45 minutes (Colab GPU)
- **Model Size**: 6.2MB (YOLOv8n)

### **Inference Performance**
- **Speed**: 45+ FPS (Colab GPU)
- **Latency**: ~22ms per image
- **Memory Usage**: <2GB GPU memory

## 🎛️ Interactive Features

### **Real-time Detection Interface**
- Drag-and-drop image upload
- Adjustable confidence threshold
- Live FPS monitoring
- Bounding box visualization

### **Performance Dashboard**
- Training metrics visualization
- Real-time performance gauges
- Historical data tracking

### **Model Information Panel**
- Supported classes display
- Model architecture details
- Usage statistics

## 🔄 Customization Options

### **Training Parameters**
```python
# Modify these variables in the notebook
EPOCHS = 50              # Training epochs
IMG_SIZE = 640          # Input image size
BATCH_SIZE = 16         # Batch size (adjust for your GPU)
CONFIDENCE = 0.25       # Detection confidence threshold
```

### **Dataset Configuration**
```python
# Customize dataset splits
train_split = 0.7       # 70% for training
val_split = 0.2         # 20% for validation
test_split = 0.1        # 10% for testing
```

## 📱 Mobile-Friendly

The Gradio interface is mobile-responsive, allowing you to:
- Access the detection interface from your phone
- Upload images directly from mobile camera
- View results in real-time
- Share the interface with others

## 🔍 Troubleshooting

### **Common Issues & Solutions**

**GPU Memory Issues:**
```python
# Reduce batch size in training cell
BATCH_SIZE = 8  # Instead of 16
```

**Dataset Not Found:**
```python
# Ensure Google Drive is mounted
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**Gradio Interface Not Loading:**
```python
# Restart runtime and re-run interface cell
# Runtime → Restart and run all
```

**Slow Training:**
```python
# Verify GPU is enabled
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 📚 Learning Resources

### **Understanding the Code**
- Each cell includes detailed comments
- Markdown explanations between code sections
- Visual outputs for better understanding

### **Extending the System**
- Add custom datasets by modifying conversion cell
- Experiment with different YOLOv8 models (s, m, l, x)
- Customize detection classes and thresholds

### **Advanced Usage**
- Export model for mobile deployment
- Integration with web applications
- Batch processing capabilities

## 🤝 Contributing

### **How to Contribute**
1. Fork the repository
2. Create a new branch for your feature
3. Test changes in Google Colab
4. Submit a pull request with clear description

### **Contribution Ideas**
- Add support for custom datasets
- Implement additional model architectures
- Create new visualization features
- Improve mobile interface
- Add batch processing capabilities

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **Google Colab** for free GPU resources
- **Gradio** for easy web interfaces
- **COCO Dataset** creators
- **PyTorch** and **OpenCV** communities

## 🆘 Support

### **Getting Help**
- **Issues**: [GitHub Issues](https://github.com/yourusername/yolov8-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/yolov8-detection-system/discussions)
- **Colab Help**: Check the "Help" menu in Colab

### **Common Questions**
- **Q**: Can I run this without GPU?
  - **A**: Yes, but training will be much slower. Use CPU runtime for testing only.
- **Q**: How much does Google Colab cost?
  - **A**: Basic usage is free. Pro version available for extended GPU access.
- **Q**: Can I use my own dataset?
  - **A**: Yes! Modify the dataset conversion section to use your data.

---

## 🎯 **Ready to start detecting?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/yolov8-detection-system/blob/main/YOLOv8_Detection_System.ipynb)

**Click above and start detecting objects in minutes!**

### **Quick Demo Flow:**
1. **Open in Colab** → 2. **Connect GPU** → 3. **Run All Cells** → 4. **Start Detecting!**

*Note: All dependencies are automatically installed in the first cell - no separate setup required!*

---

**Happy Detecting! 🚀📱🎯**
