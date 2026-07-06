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

## 📂 Project Structure & File Descriptions

| File | Description |
| :--- | :--- |
| **`anpr_colab.py`** | The primary AI pipeline that handles vehicle detection, ByteTrack tracking, PaddleOCR plate reading, and API communication. |
| **`plate_rules.py`** | Contains specialized domain logic to validate and auto-correct Indian license plate OCR reads using character-mapping and regex. |
| **`main.py`** | The FastAPI entry point that initializes the database, mounts static assets, and hosts the API router. |
| **`routes.py`** | The central API router that consolidates all endpoints for detections, analytics, and manual logging. |
| **`db.py`** | Manages the SQLAlchemy database engine, session factory, and SQLite connection configuration. |
| **`models.py`** | Defines the database schema for detection logs using SQLAlchemy ORM. |
| **`crud.py`** | Handles all database operations including searching plates, creating detection logs, and calculating dashboard statistics. |
| **`dashboard.py`** | The Streamlit-powered Security Operations Center (SOC) dashboard that visualizes live traffic telemetry and detection logs. |
| **`config.py`** | A centralized configuration file managing file paths, API URLs, and hardware device settings. |

## 🖥️ Hardware Requirements
This system performs real-time multi-model inference (YOLO + PaddleOCR). To achieve smooth performance:
* **Recommended:** 16/32GB+ RAM and a dedicated NVIDIA GPU (CUDA enabled).
* **Alternative:** If running on a local CPU, the processing speed will decrease significantly.
* **Cloud Option:** Highly recommended to run on [Google Colab](https://colab.research.google.com/) or similar cloud environments with T4/L4 GPU acceleration.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install all required dependencies listed in `requirements.txt`:
```bash
git clone [https://github.com/bhavyabundela06/Indian-ANPR-ByteTrack-FastAPI.git](https://github.com/bhavyabundela06/Indian-ANPR-ByteTrack-FastAPI.git)
cd Indian-ANPR-ByteTrack-FastAPI
pip install -r requirements.txt

*Configure Your Environment
•	Download your custom plate detection model and place it in the models/ folder as best.pt.
•	Ensure yolov8n.pt is also present in the models/ folder.
•	Place your test video in the videos/ folder.

*Run the System
1.	Backend: uvicorn main:app --reload
2.	AI Pipeline: python anpr_colab.py
3.	Dashboard: streamlit run dashboard.py

📊 Dataset & Training
The object detection models utilized in this pipeline were trained and evaluated using high-quality Indian traffic data sourced from Kaggle.
•	Dataset Link: [Insert your Kaggle dataset link here]

***


