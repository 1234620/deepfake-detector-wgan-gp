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

## Features

- WGAN-GP based deepfake detection training pipeline.
- Flask backend with image upload inference endpoint.
- Front page for live prediction and separate report page for evaluation plots.
- Saved checkpoints and threshold-based critic inference.

## Repository Structure

- `backend/` Flask server and inference API.
- `frontend/` inference page, report page, scripts, and styles.
- `model/` training loop, architecture, and hyperparameters.
- `checkpoints/` trained weights and threshold file.
- `results/` confusion matrix, ROC curve, and training loss curves.

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
