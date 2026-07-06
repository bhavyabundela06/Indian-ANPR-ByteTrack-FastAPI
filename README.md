# Indian-ANPR-ByteTrack-FastAPI 🛡️

An end-to-end, real-time Automatic Number Plate Recognition (ANPR) system engineered specifically for Indian traffic domain rules. 

Most open-source ANPR scripts freeze or drop frames when processing heavy OCR tasks. This project solves that by utilizing a completely decoupled, multithreaded architecture. The video display, YOLO detection, PaddleOCR processing, and database logging all run asynchronously, ensuring smooth performance even on local hardware. 

Coupled with a FastAPI backend and a Streamlit Security Operations Center (SOC) dashboard, this system goes beyond a simple Python script to deliver a full-stack traffic enforcement solution.

## ⚡ Key Features
* **Real-Time Tracking:** Utilizes YOLOv8 and ByteTrack to lock onto cars, bikes, buses, and trucks across multiple frames without losing the subject.
* **Domain-Aware OCR:** Implements custom regex and character-mapping rules specific to the Indian format (`SS DD L NNNN`). It automatically corrects common AI hallucinations (e.g., confusing `O` with `0` in a digit-only slot).
* **Decoupled Architecture:** The display loop never blocks on heavy OCR processing, allowing the video stream to maintain high FPS. 
* **Full-Stack Integration:** Includes a FastAPI backend router and a lightweight SQLite database to log detections, confidence scores, and timestamps.
* **Command Matrix Dashboard:** A sleek, live-updating Streamlit SOC interface to monitor traffic telemetry and view evidence crops in real-time.

## 🗂️ System Architecture
* `anpr_colab.py`: The core computer vision pipeline (YOLOv8 + PaddleOCR + ByteTrack).
* `plate_rules.py`: Regex and algorithmic correction for Indian license plate formats.
* `main.py` / `routes.py`: The FastAPI application and endpoint routing.
* `db.py` / `models.py` / `crud.py`: SQLite database engine and transaction logic.
* `dashboard.py`: The Streamlit-powered Command Matrix frontend.

## 🖥️ Hardware Requirements
This system performs real-time multi-model inference (YOLO + PaddleOCR). To achieve smooth performance:
* **Recommended:** 16/32GB+ RAM and a dedicated NVIDIA GPU (CUDA enabled).
* **Alternative:** If running on a local CPU, the processing speed will decrease significantly.
* **Cloud Option:** Highly recommended to run on [Google Colab](https://colab.research.google.com/) or similar cloud environments with T4/L4 GPU acceleration.

## 🚀 Getting Started

### 2. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install fastapi uvicorn streamlit sqlalchemy ultralytics paddleocr opencv-python pandas requests