
# HemaVision — CRT & Pallor Vision Prototype (OpenCV)
OpenCV-based system that analyzes fingertip redness and return thresholds to measure CRT and detect pallor, designed for future integration with embedded hardware (Raspberry Pi).
> Research prototype for educational use only — **not a medical device**.

## Features
- ROI-based redness signal with baseline → press → release **state machine**
- Moving-average smoothing; adaptive return threshold (e.g., 95% of baseline)
- On-screen guidance & telemetry; optional live plotting for debugging
- Designed to run on laptop webcam; **hardware integration (Raspberry Pi) planned**

## Install
```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
