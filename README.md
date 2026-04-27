# 🌿 Smart Green POT — Automated Plant Care System

An intelligent plant care system that combines **deep learning-based disease detection** with **real-time closed-loop environmental control** to minimise manual plant maintenance.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `Train_Plant_Disease.ipynb` | CNN model training for plant disease classification |
| `YoloTrain.ipynb` | YOLOv8 training on Google Colab for leaf detection |
| `Webcam.py` | Real-time leaf detection using YOLOv8 + OpenCV |
| `test.py` | Real-time disease classification via webcam |
| `test2.py` | Disease detection from an uploaded image |

---

## 🧠 Train_Plant_Disease.ipynb — CNN Disease Classifier

A custom CNN trained from scratch to classify plant leaf diseases across 38 classes.

**Dataset:**
- Training set: **70,295 images** across 38 classes (Kaggle)
- Validation set: **17,572 images** across 38 classes (Kaggle)
- Plant species: Tomato, Apple, Cherry, Corn, Grape, Strawberry, Pepper
- Example classes: `Tomato_healthy`, `Tomato_Bacterial_spot`, `Grape_Leaf_blight`

**Architecture:**
- Two `Conv2D` layers with varying filter sizes to learn textures, edges, and patterns
- Output: 3D tensor of shape `(128, 128, num_filters)`
- `MaxPool2D` layer to select strongest neuron activations and prevent overfitting
- Learning rate: `0.0001` (to avoid overshooting)

**Results (50 epochs):**
- ✅ Training Accuracy: **94%**
- ✅ Validation Accuracy: **90%**

**CNN Architecture:**

<img src="https://github.com/user-attachments/assets/49ccb790-8ddd-4867-8536-f7af5097753e" alt="CNN Layer" width="400"/>

**Accuracy Visualization:**

<img src="https://github.com/user-attachments/assets/57eb61e5-f970-4a38-b49f-6139ad054a2d" alt="Accuracy Graph" width="400"/>

<img src="https://github.com/user-attachments/assets/4803a8ec-f9a4-4ce1-91ea-d2c45c0410a6" alt="Accuracy Graph 50 epochs" width="400"/>

---

## 📷 Webcam.py — Real-time Leaf Detection (YOLOv8)

Uses a custom-trained YOLOv8 model to detect and localise tomato leaves in real time via webcam, drawing bounding boxes around detected leaves.

**Image uploaded detection:**

<img src="https://github.com/user-attachments/assets/c095f14d-55b3-439c-ac73-a82c16f81aac" alt="Image Detection" width="400"/>

**Real-time webcam detection:**

<img src="https://github.com/user-attachments/assets/9a9eb9de-6d8c-4923-9537-22300bebb5c9" alt="Webcam Detection" width="400"/>

```bash
pip install -U ultralytics
```

---

## 🏋️ YoloTrain.ipynb — YOLOv8 Training (Google Colab)

Trains a YOLOv8 model on a custom tomato leaf dataset using Google Colab's GPU. Outputs model weights that can be used directly in `Webcam.py`.

---

## 🔍 test.py — Real-time Disease Classification

Captures live video from the webcam and classifies whether the detected plant leaf is diseased or healthy. Result is displayed as a labelled green box around the leaf.

<img src="https://github.com/user-attachments/assets/c59737f4-e285-42b3-8ded-7b0a9c3fce1f" alt="Real-time Classification" width="400"/>

```bash
pip install tensorflow
```

---

## 🖼️ test2.py — Image-based Disease Detection

Classifies disease from a single uploaded image using the trained CNN model.

🔗 [Sample model weights (Google Drive)](https://drive.google.com/file/d/1VuoScpUIpkugshXDCpRj5RnPwURXQL7o/view?usp=sharing)

---

## 🛠️ Tech Stack

| Component | Tools |
|-----------|-------|
| Disease Classification | TensorFlow, Keras, CNN |
| Leaf Detection | YOLOv8, OpenCV, Ultralytics |
| Mobile Deployment | Flutter, Hugging Face Inference API |
| Environmental Control | LabVIEW, NI-DAQmx |
| Data & Visualisation | Pandas, Matplotlib, Seaborn |

---

## 🚀 Getting Started

```bash
pip install tensorflow ultralytics opencv-python matplotlib pandas seaborn
```
