# 🎭 Robust Face Recognition with CASIA & Occluded Datasets

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
| Occluded     | 200      | 1600   | Left crop, right Crop,  Low brightness, high brightness, Blackdots, Blurred, Sktech,      | 640x480    |
                                  


## 🛠 Technical Implementation
###Core Components
  - Face Processing
  - MTCNN-based alignment with 5-point landmarks
  -Adaptive histogram equalization
  -160x160 normalized RGB output

###Feature Engineering

-FaceNet-512 embeddings

-PCA dimensionality reduction (512 → 128)

-L2-normalized feature vectors
