# Deepfake Detector (WGAN-GP)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Last Commit](https://img.shields.io/github/last-commit/1234620/deepfake-detector-wgan-gp?style=for-the-badge&logo=github)](https://github.com/1234620/deepfake-detector-wgan-gp/commits)
[![Repo Size](https://img.shields.io/github/repo-size/1234620/deepfake-detector-wgan-gp?style=for-the-badge)](https://github.com/1234620/deepfake-detector-wgan-gp)
[![Stars](https://img.shields.io/github/stars/1234620/deepfake-detector-wgan-gp?style=for-the-badge)](https://github.com/1234620/deepfake-detector-wgan-gp/stargazers)

Deepfake detection pipeline built on WGAN-GP with a runnable Flask backend, an image upload inference homepage, and a separate report page.

## What This Repository Contains

- End-to-end training pipeline for deepfake detection
- Saved checkpoints and threshold for inference
- Web backend API for prediction
- Clean frontend inference page
- Separate report page for evaluation visuals

## Project Layout

- `model/` training code, architecture, losses, and configuration
- `checkpoints/` model weights and threshold
- `results/` confusion matrix, ROC curve, training loss plot
- `backend/` Flask server for API and frontend routes
- `frontend/` inference UI and report page

## Run Locally

Install dependencies:

```bash
cd "/Users/ahmedmoosani/Downloads/1st Model"
python3 -m pip install -r model/requirements.txt
python3 -m pip install flask
```

Start the app:

```bash
bash frontend/start_frontend.sh
```

Open in browser:

- Home: `http://127.0.0.1:5500/`
- Report: `http://127.0.0.1:5500/report`

If port 5500 is already used:

```bash
PID=$(lsof -ti :5500 || true)
[ -n "$PID" ] && kill "$PID"
bash frontend/start_frontend.sh
```

## API Endpoints

Health:

```bash
curl http://127.0.0.1:5500/api/health
```

Predict from image:

```bash
curl -X POST http://127.0.0.1:5500/api/predict -F "image=@/path/to/image.jpg"
```

## Inference Details

- Inference model: `checkpoints/critic_final.pth`
- Threshold source: `checkpoints/threshold.txt`
- Preprocessing: resize to 64x64, normalize to [-1, 1]
- Decision rule: score > threshold => REAL, else DEEPFAKE

## Repository

- URL: https://github.com/1234620/deepfake-detector-wgan-gp
