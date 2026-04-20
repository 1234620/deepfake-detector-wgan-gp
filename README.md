<div align="center">

# DEEPFAKE DETECTION SYSTEM

### *Detecting Manipulation, Preserving Truth, Securing Authenticity*

[![Last Commit](https://img.shields.io/github/last-commit/1234620/deepfake-detector-wgan-gp?style=flat-square&label=last%20commit)](https://github.com/1234620/deepfake-detector-wgan-gp/commits)
[![Top Language](https://img.shields.io/github/languages/top/1234620/deepfake-detector-wgan-gp?style=flat-square&label=top%20language)](https://github.com/1234620/deepfake-detector-wgan-gp)
[![Languages](https://img.shields.io/github/languages/count/1234620/deepfake-detector-wgan-gp?style=flat-square&label=languages)](https://github.com/1234620/deepfake-detector-wgan-gp)

### *Built with the tools and technologies:*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/docs/Web/JavaScript)

</div>

---

## Project Overview

This project builds a deepfake detection system using a WGAN-GP training pipeline.
During training, a Generator creates synthetic images and a Critic learns to score image
realness. After training, the Generator is discarded and the Critic is reused as the
detector model.

The repository includes a runnable backend and frontend:

- The frontend home page accepts an image upload.
- The backend runs inference with the trained Critic checkpoint.
- The model score is compared against a saved threshold.
- The output is returned as REAL or DEEPFAKE.
- A separate report page displays saved evaluation artifacts.

## Features

- WGAN-GP based deepfake detection training pipeline.
- Flask backend with image upload inference endpoint.
- Front page for live prediction and separate report page for evaluation plots.
- Saved checkpoints and threshold-based critic inference.

## Project Tree

```text
1st Model/
├── backend/
│   └── app.py
├── checkpoints/
│   ├── critic_final.pth
│   ├── generator_final.pth
│   ├── threshold.txt
│   ├── crit_epoch_*.pth
│   └── gen_epoch_*.pth
├── frontend/
│   ├── index.html
│   ├── report.html
│   ├── main.js
│   ├── app.js
│   ├── main.css
│   ├── styles.css
│   └── start_frontend.sh
├── model/
│   ├── config.py
│   ├── models.py
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_losses.png
└── README.md
```

## What The System Does

1. Accepts an input image from the frontend or API.
2. Applies the same preprocessing used in model training (resize and normalization).
3. Runs the image through the trained Critic network.
4. Compares the score with the saved threshold.
5. Returns a final binary classification: REAL or DEEPFAKE.

## Quick Start

Install dependencies:

```bash
cd "/Users/ahmedmoosani/Downloads/1st Model"
python3 -m pip install -r model/requirements.txt
python3 -m pip install flask
```

Run backend and frontend:

```bash
bash frontend/start_frontend.sh
```

Open in browser:

- Home: `http://127.0.0.1:5500/`
- Report: `http://127.0.0.1:5500/report`

If port 5500 is already in use:

```bash
PID=$(lsof -ti :5500 || true)
[ -n "$PID" ] && kill "$PID"
bash frontend/start_frontend.sh
```

## API

Health:

```bash
curl http://127.0.0.1:5500/api/health
```

Predict:

```bash
curl -X POST http://127.0.0.1:5500/api/predict -F "image=@/path/to/image.jpg"
```

## Inference Runtime

- Model weights: `checkpoints/critic_final.pth`
- Threshold file: `checkpoints/threshold.txt`
- Input preprocessing: resize to 64x64 and normalize to [-1, 1]
- Decision rule: score > threshold -> REAL, otherwise DEEPFAKE

## Repository Link

https://github.com/1234620/deepfake-detector-wgan-gp
