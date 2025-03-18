# 🎭 Face Recognition System with CASIA & Occluded Datasets

[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-green)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![Face Recognition Demo](demo.gif) *Sample recognition under occlusion*

## 🔍 Overview
A cutting-edge face recognition system evaluating model robustness against real-world challenges. Leverages **FaceNet embeddings** and **ensemble machine learning** to achieve state-of-the-art performance on both pristine and occluded facial images.

## 📂 Dataset Architecture
### **Core Datasets**
| Dataset      | Subjects | Images | Occlusion Types                                                                           | Resolution |
|--------------|----------|--------|-------------------------------------------------------------------------------------------|------------|
| CASIA        | 14       | 190    | None                                                                                      | 640x480    |
| Occluded     | 200      | 1600   | Left crop, right Crop,  Low brightness, high brightness, Blackdots, Blurred and Sktech    | 640x480    |

### **Custom Occlusion Pipeline**
I generated challenging synthetic variations from CASIA baseline using:
- **Spatial Transformations**  
  🌄 `Left/Right Crop` (20-30% region removal)  
  🖼️ `Sketch Effect` (Canny edge detection + stylization)
  
- **Photometric Distortions**  
  🔆 `Low/High Brightness` (±50% delta)  
  ⚫ `Blackdots Noise` (5-15% coverage)  
  🌫️ `Gaussian Blur` (σ=1.5-3.0)

*Example augmentation pipeline:*
```python
def apply_occlusion(img):
    occlusion_type = random.choice([
        'left_crop', 'right_crop', 
        'low_light', 'overexposure',
        'blackdots', 'gaussian_blur',
        'pencil_sketch'
    ])
    # Implementation details in augmentation.py
    return transformed_image
```


## 🛠 Technical Implementation

### Core Components
   - Face Processing
   - MTCNN-based alignment with 5-point landmarks
   - Adaptive histogram equalization
   - 160x160 normalized RGB output

### Feature Engineering
  - FaceNet-512 embeddings
  - PCA dimensionality reduction (512 → 128)
  - L2-normalized feature vectors

## 📊 Performance Analysis

| Condition    | Accuracy  | Precision | Recall      | Support    |
|--------------|---------- |--------   |-------------|------------|
| CASIA        |    95%    |    95%    |      94%    |    55      |
| Occluded     |    98%    |    98%    |      97%    |    370     |

## 🚀 Getting Started
### Installation
```python
git clone https://github.com/yourusername/robust-face-recognition.git
conda create -n facerec python=3.8
conda activate facerec
pip install -r requirements.txt

```
### Dataset Setup
```python
# unzip data.zip
# Download CASIA-Web dataset and Occluded dataset to respective folders
```
### Training & Evaluation
``` python
jupyter notebook
# 1. Face Recognition using CASIA dataset.ipynb - Base model training
# 2. Face Recognition using Occluded dataset.ipynb - Robustness evaluation
```
## 🌟 Future Roadmap
   - Dynamic Occlusion Handling
   - Add LFW and MegaFace benchmarks
   - TensorFlow Lite for ESP32-CAM integration
   - 3D Face Reconstruction


