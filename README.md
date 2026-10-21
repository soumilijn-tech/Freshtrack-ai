---
title: FreshTrack AI
emoji: 🍃
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# FreshTrack AI — Smart Fruit & Vegetable Freshness Assistant 🍃

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/AI-PyTorch%20Transfer%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Database](https://img.shields.io/badge/Database-SQLite%203-003B57.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-14%20Passing%20(100%25)-brightgreen.svg)]()

FreshTrack AI is an end-to-end, mobile-first AI web application developed as a job-ready **Computer Science & Artificial Intelligence / Machine Learning (CSE-AIML)** portfolio project. It enables users to snap or upload produce photos, classify fruits and vegetables, estimate visible freshness with calibrated uncertainty metrics, look up scientific storage advice, and manage household kitchen inventory to reduce domestic food waste.

---

## 📸 Key Features

1. **Intelligent Produce Scanner**:
   - Camera-friendly mobile interface with native capture and live webcam streamer.
   - 1-Click instant test samples (Fresh Apple, Spotted Banana, Fresh Tomato, Fresh Carrot).
2. **Dual-Head AI Architecture**:
   - Classifies 14 produce classes (Apple, Banana, Orange, Tomato, Potato, Carrot, Bell Pepper, Broccoli, Strawberry, Cucumber, Onion, Lettuce, Avocado, Lemon).
   - Detects 3 visible freshness levels (`Fresh`, `Fair / Consume Soon`, `Overripe / Spoiled`).
3. **Calibrated Confidence & Uncertainty**:
   - Displays Softmax class probabilities and Shannon entropy-based uncertainty scores.
   - Transparently flags ambiguous or low-confidence captures.
4. **Food Safety & Ethical AI Notice**:
   - Prominently clarifies that camera images only judge surface optical decay; bacterial toxins and anaerobic pathogens cannot be evaluated by photo alone.
5. **Science-Backed Storage Advisor**:
   - Encyclopedia of 14+ fruits and vegetables with optimal temperature, humidity, ethylene production/sensitivity profiles, and practical kitchen preservation hacks.
6. **Kitchen Inventory & Expiry Tracker**:
   - Automatic pre-fill of scanned items into inventory with dynamic use-by dates.
   - Status indicators (`Fresh`, `Expiring Soon`, `Expired`, `Consumed`, `Discarded`).
   - One-click actions to mark items as consumed (saved from waste) or discarded.
7. **Food Waste Analytics Dashboard**:
   - Interactive Chart.js visualizations (Freshness distribution doughnut, storage locations bar chart).
   - Real-time KPIs: Active items, Expiring soon count, Waste Prevention Rate (%), and Estimated Dollar Value saved.
8. **Export & Portability**:
   - Single-click CSV and JSON inventory export.

---

## 🏛️ System Architecture

```
                                  [ User Device ]
                       Mobile Smartphone / Desktop Browser
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Frontend UI (Vanilla JS)  │
                       │  • Camera capture & preview   │
                       │  • Freshness gauge & charts   │
                       │  • Inventory & Waste tracker  │
                       └───────────────┬───────────────┘
                                       │ HTTP / REST & Multipart
                                       ▼
                       ┌───────────────────────────────┐
                       │        FastAPI Backend        │
                       │   Router, Validation & ORM    │
                       └───────┬───────────────┬───────┘
                               │               │
            ┌──────────────────┴──┐         ┌──┴──────────────────┐
            ▼                     ▼         ▼                     ▼
┌───────────────────────┐ ┌───────────────┐ ┌───────────────────┐ ┌─────────────┐
│  VisionModelService   │ │Storage Advisor│ │ Inventory Engine  │ │  SQLite DB  │
│ • PyTorch Model       │ │• Storage tips │ │• Expiry & status  │ │• Items      │
│ • Demo CV Heuristic   │ │• Ethylene gas │ │• Consumed/Wasted  │ │• Scans      │
│ • Entropy Uncertainty │ │• Shelf-life   │ │• CSV/JSON Export  │ │• Waste logs │
└───────────────────────┘ └───────────────┘ └───────────────────┘ └─────────────┘
```

---

## ⚡ Quickstart Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, and 3.13)
- Modern web browser (Chrome, Safari, Firefox, Edge)

### 1. Clone or Navigate to the Project
```bash
cd C:\Users\LENOVO\.gemini\antigravity\scratch\FreshTrackAI
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run the Backend Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Web Application
Open your browser and navigate to:
- **Web App**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)
- **ReDoc API Documentation**: [http://127.0.0.1:8000/api/redoc](http://127.0.0.1:8000/api/redoc)

---

## 🤖 AI Model: Demo Mode vs. Trained PyTorch Model

To uphold academic and engineering integrity, FreshTrack AI **never pretends a model is trained if no trained weights exist**:

1. **Demo Mode (Default)**:
   - When no `freshtrack_model.pth` exists in `backend/models/`, the backend safely runs in **DEMO MODE**.
   - It is clearly labeled with an amber banner in the UI and in every API response (`mode: "demo_heuristic"`).
   - In Demo Mode, it analyzes image RGB, hue dominance, and dark/brown texture spots to generate realistic predictions.
2. **Active PyTorch Model**:
   - Once you train the model in Google Colab and place `freshtrack_model.pth` into `backend/models/`, the server detects it on startup, transitions the badge to green **PyTorch Model Active**, and runs neural network tensor inference.

---

## 🔬 Training in Google Colab

A production-ready Colab training notebook is provided at:
`notebooks/FreshTrack_AI_Training_TransferLearning.ipynb`

### Steps to Train:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `FreshTrack_AI_Training_TransferLearning.ipynb`.
3. In Colab, navigate to **Runtime > Change runtime type > T4 GPU**.
4. Run all cells:
   - Sets up data augmentations (RandomResizedCrop, ColorJitter, Rotation).
   - Loads MobileNetV2 pretrained on ImageNet.
   - Fine-tunes dual heads (Produce head + Freshness head) with AdamW optimizer.
   - Evaluates with confusion matrix and Shannon entropy uncertainty calculation.
   - Exports `freshtrack_model.pth`.
5. Download `freshtrack_model.pth` and place it in:
   ```
   FreshTrackAI/backend/models/freshtrack_model.pth
   ```
6. Restart the backend. FreshTrack AI will automatically switch to active PyTorch model mode!

---

## 🧪 Running Unit Tests

FreshTrack AI includes an automated test suite verifying all API endpoints, database operations, classification heuristics, and export formats:

```bash
python -m pytest backend/tests -v
```

**Results**: 14 passing tests covering:
- Vision service initialization & demo mode fallback
- Probability distribution integrity and safety disclaimers
- Storage advisor recommendations & shelf-life calculation
- Inventory CRUD, filtering, search, and action logging
- CSV and JSON export endpoints
- Dashboard food waste statistics and KPI calculations

---

## 🚀 Free-Tier Deployment Options

### Option 1: Render (Free Web Service)
1. Fork or push this repository to GitHub.
2. Sign in to [Render.com](https://render.com).
3. Click **New + > Web Service** and connect your GitHub repo.
4. Set:
   - **Environment**: `Python 3` (or `Docker` using the included Dockerfile)
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**. Your app is live with free SSL!

### Option 2: Hugging Face Spaces (Free Docker Space)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Docker** SDK.
3. Push this repository. Hugging Face will automatically build the container from `Dockerfile` and expose it globally!

### Option 3: Local Docker Container
```bash
docker build -t freshtrack-ai .
docker run -p 8000:8000 freshtrack-ai
```

---

## 🛡️ Food Safety Disclaimer
> FreshTrack AI estimates visible freshness based on optical imaging features (surface discoloration, blemishes, spotting). Microorganisms, anaerobic pathogens (such as *Salmonella*, *Listeria*, or *E. coli*), and internal decay cannot be detected from photos. Always inspect produce smell, texture, and internal flesh before preparation or consumption.

---

## 📜 License & Portfolio Attribution
Developed as an advanced CSE-AIML portfolio project demonstrating full-stack engineering, transfer learning computer vision, calibrated uncertainty modeling, and practical environmental food waste mitigation.
