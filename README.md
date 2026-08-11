# FabMetricsAi — Semiconductor Wafer Defect Detection System

WaferScan AI is an industrial-grade computer vision platform built for automated microchip quality control and semiconductor cleanroom yield analysis. It leverages a fine-tuned **ResNet-34** deep residual network to classify silicon wafer map defect patterns (Scratch, Donut, Edge-Ring, Edge-Loc, Loc, Center, Near-full, Random, None) with **87.51% Macro-F1** accuracy.

---

## Key Features

- 🔬 **High-Precision Classification**: Classifies wafer maps across 9 failure modes matching the WM-811K benchmark.
- ⚡ **Real-Time Inference**: Sub-15ms prediction latency powered by PyTorch ResNet-34.
- 🎯 **Automated Defect Segmentation**: Computer Vision (OpenCV) contour isolation highlights defect locations with overlay markers.
- 📄 **Executive PDF Yield Reports**: Automated ReportLab PDF generation for batch processing results.
- 🎨 **Interactive UI & Custom Themes**: Modern dark-mode dashboard featuring 55 custom themes, deep analytics benchmarks, and dataset showroom.
- 🤖 **Cleanroom AI Tutor**: Embedded domain assistant for cleanroom protocols and defect physics.

---

## Project Structure

```text
├── app.py                     # Main FastAPI Server & Application Routing
├── wafer_dataset.py           # PyTorch Dataset Loader & Preprocessing Pipeline
├── train.py                   # Model Training Script with Weighted Sampler
├── test_inference.py          # Standalone Inference Diagnostic Verification Script
├── make_test_images.py        # Synthetic Evaluation Wafer Map Generator
├── baseline_resnet34.pth      # Trained ResNet-34 Model Checkpoint Weights
├── requirements.txt           # Python Dependencies File
├── inference/                 # Modular Python Inference Package
│   ├── config.py              # Class Definitions & Benchmark Metrics
│   ├── model.py               # ResNet-34 Architecture Construction & Loaders
│   └── preprocess.py          # Input Decoding & CV Defect Localization
├── frontend/                  # Web Dashboard Application
│   ├── index.html             # Main Dashboard User Interface
│   ├── themes.js              # Theme Gallery & Custom Color Engine
│   ├── chatbot.js             # Cleanroom AI Tutor Assistant Module
│   └── app.js                 # Alternate Dashboard Logic Component
└── test_samples/              # Folder for Test Images & Evaluation Substrates
```

---

## Quick Start Guide

### 1. Install Dependencies

Ensure Python 3.9+ is installed, then install required packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Application Server

Start the FastAPI server directly:

```bash
python app.py
```

Or using uvicorn:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 `http://127.0.0.1:8000/`

---

## Generating Test Data & Diagnostic Verification

### Generate Synthetic Test Images

To generate 10 test wafer map PNGs with synthetic defect patterns:

```bash
python make_test_images.py
```

This populates the `test_samples/` directory. Upload these files to the dashboard queue to test batch execution and PDF generation.

### Run Diagnostic Verification

To test local weight loading and tensor inference pipeline:

```bash
python test_inference.py
```

---

## API Documentation

- `GET /` — Serves the interactive frontend console.
- `GET /model-info` / `GET /api/health` — Returns backend operational status, active device (CUDA/CPU), and model architecture info.
- `GET /api/baselines` — Returns comparative benchmark Macro-F1 metrics for ResNet-34, Shallow CNN, and HOG+RF.
- `POST /predict` — Upload an image file (`file`) for inference and defect segmentation overlay.
- `POST /api/predict` — Accepts JSON payload with `wafer_map` matrix array for classification.
- `POST /generate-report` — Generates a downloadable PDF report summarizing batch inspection results.

---

## License & Credits

Engineered by Chitransh Saxena & Team.
Dataset reference: WM-811K Wafer Map Dataset.
