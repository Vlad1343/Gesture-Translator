# British Sign Language Translator

> 🤟 Real-Time Computer Vision & Sequence Learning Project  
> ♿ Accessibility-Focused AI System for Gesture Understanding  

---

## 🚀 Overview

**British Sign Language Translator** is a real-time, AI-powered computer vision system that translates **British Sign Language (BSL) gestures into plain English text** using webcam input. The system captures human motion via MediaPipe, extracts spatiotemporal keypoints from hands, face, and upper body, and performs sequence-based gesture classification using a deep learning model trained on recorded sign sequences.

Designed with **accessibility, low latency, and modularity** in mind, the project enables intuitive, camera-based sign recognition that runs locally and responds instantly, helping bridge communication gaps between signers and non-signers.

---

## 💡 Core Features

### 🤲 Real-Time Gesture Recognition

- Live webcam capture using **OpenCV** with frame-by-frame landmark tracking.
- MediaPipe **Holistic** model tracks hands, face, and upper body simultaneously.
- Robust to natural motion variation and differences in signing speed.

### 🧠 Temporal Gesture Understanding

- Gestures represented as **time-based sequences of keypoints**, preserving motion dynamics.
- Fixed-length temporal windows enable consistent training and inference.
- Sliding window inference allows continuous recognition without manual segmentation.

### 🧮 Deep Learning Classifier

- **LSTM-based neural network** trained on sequential keypoint data.
- Learns temporal dependencies unique to each sign.
- Outputs probabilistic predictions mapped directly to plain-English labels.

### 🖥️ Instant Feedback Loop

- As soon as a gesture is recognized with sufficient confidence, it is printed to the terminal.
- Low-latency, on-device inference without reliance on cloud services.
- Designed for real-time experimentation and live demonstrations.

### ♿ Accessibility-Oriented Design

- Camera-only interaction — no gloves, trackers, or external sensors required.
- Runs fully locally for privacy, reliability, and offline use.
- Simple workflow for extending the vocabulary with new gestures.

---

## 🏗️ System Architecture

Webcam  
→ MediaPipe Holistic  
→ Keypoint Extraction (Hands + Face + Pose)  
→ Temporal Sequence Buffer  
→ LSTM Gesture Classifier  
→ English Text Output

- **Webcam:** Captures live video for gesture recognition.  
- **MediaPipe Holistic:** Detects 3D landmarks for hands, face, and upper body.  
- **Keypoint Extraction:** Converts landmarks into normalized numerical sequences.  
- **Temporal Sequence Buffer:** Maintains recent frames to provide temporal context for classification.  
- **LSTM Gesture Classifier:** Learns motion patterns and predicts gestures.  
- **English Text Output:** Displays recognized gestures in real time.  

---

## 🏗️ Tech Stack

| Layer | Technologies |
|------|--------------|
| Computer Vision | OpenCV, MediaPipe Holistic |
| Machine Learning | PyTorch, LSTM |
| Data Processing | NumPy |
| Runtime | Python 3.11 |
| Deployment | Local, offline-first |

---

## 🧠 Why This Project Stands Out

✅ End-to-end ML system covering **data capture, training, and real-time inference**  
✅ Sequence-based learning instead of static pose classification  
✅ Multi-landmark fusion using hands, face, and body keypoints  
✅ Accessibility-driven application with clear real-world relevance  
✅ Clean, modular architecture suitable for extension and research  

This project demonstrates proficiency in **computer vision pipelines**, **temporal deep learning models**, and **practical AI system design**.

---

## 🔮 Future Enhancements

- Multi-gesture sentence recognition with automatic segmentation  
- Temporal attention mechanisms for improved classification accuracy  
- Web-based demo using real-time video streaming  
- Mobile deployment with on-device inference  
- Text-to-speech output for bi-directional communication  

---

## 🎯 Example Use Case

1. User stands in front of a webcam.  
2. Performs a BSL gesture (e.g. *“yes”*).  
3. System captures motion across multiple frames.  
4. Model classifies the gesture in real time.  
5. The recognized word is displayed instantly in English.

---

**British Sign Language Translator** demonstrates how modern computer vision and deep learning can be combined into a **real-time, accessible, and impactful AI application**, transforming human motion into language with speed, precision, and purpose.
