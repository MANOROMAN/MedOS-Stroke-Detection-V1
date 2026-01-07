<div align="center">
  <br />
  <h1>MedOS Suite v1.0</h1>
  <h3>Next-Generation Neuro-Diagnostic System (NDS) | Automated Stroke Detection</h3>
  <br />
  <p>
    <a href="#about">About</a> &nbsp;•&nbsp;
    <a href="#features">Features</a> &nbsp;•&nbsp;
    <a href="#installation">Installation</a> &nbsp;•&nbsp;
    <a href="#technical-architecture">Technical Specs</a>
  </p>
  <br />
</div>

---

## About
**MedOS Suite** is a specialized Picture Archiving and Communication System (PACS) module designed to assist radiologists and neurologists in the early detection and segmentation of ischemic stroke lesions. Powered by a multi-architecture Deep Learning ensemble, it reduces diagnostic workload and offers a robust "Second Opinion" in critical scenarios.

Going beyond traditional viewers, MedOS integrates real-time inference directly into the clinical workflow, offering seamless ROI analysis, risk scoring, and automated reporting.

---

## Features

### Multi-Modal Inference Engine
An intelligent ensemble of three specialized architectures for maximized sensitivity:
- **DeepLabV3+ (EfficientNet-B5):** High-precision segmentation for complex lesion shapes.
- **Attention U-Net:** Focuses on subtle, small ischemic regions often missed by standard convolutions.
- **YOLO11:** Rapid object detection for immediate region flagging.
- **Ensemble Logic:** Aggregates predictions to ensure robust and reliable diagnostic suggestions.

### Interactive Diagnostic Viewer
A professional-grade interface designed for speed and clarity:
- **Smart ROI Analysis:** Crop and re-analyze specific suspicious regions for granular risk assessment.
- **Real-Time Visualization:** Toggle between filled masks, outlines, or high-contrast modes.
- **Traceability Matrix:** Session-based history stack allowing instant comparison of multiple views.

### Audit & Reporting
- **One-Click Reporting:** Generate instant snapshot reports with risk metrics.
- **Session History:** Tracks every analysis performed during the session.
- **Secure Handling:** Data is processed locally without external cloud dependencies.

---

## Installation

Follow these steps to run the project locally.

### 1. Repository Setup
Extract the project files to your preferred directory.
```bash
cd MedOS-Suite
```

### 2. Install Dependencies
Install the required Python packages using pip.
```bash
pip install -r requirements.txt
```

### 3. Model Configuration (Critical)
Due to their size, model weights are hosted separately. You must place the following files into the `models/` directory:
- `best_deeplabv3_final.pth`
- `best_attention_unet.pth`
- `best_yolo11.pt`

### 4. Launch Application
Start the inference server.
```bash
python app.py
```
*Access the interface at:* `http://localhost:5000`

---

## Technical Architecture

### Backend (Inference Node)
- **Framework:** Python / Flask
- **Core:** PyTorch, Ultralytics, Albumentations
- **Operation:** RESTful API handling multipart image streams.

### Frontend (Client)
- **Framework:** Vanilla JavaScript (ES6+)
- **Rendering:** HTML5 Canvas for zero-latency manipulation.
- **Design:** Optimized for high-contrast radiology environments.

---

## Disclaimer
This software is for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**. It is not FDA/CE approved for clinical diagnosis.

<br />
<div align="center">
  <p>Developed by <strong>Yusuf GÜNEL</strong></p>
  <a href="https://github.com/MANOROMAN">
    <img src="https://img.shields.io/badge/GitHub-MANOROMAN-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
</div>
