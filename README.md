# Deepfake Detector (WGAN-GP)

![Python](https://img.shields.io/badge/Python-3.10%2B-111111?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-111111?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.x-111111?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-111111?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/1234620/deepfake-detector-wgan-gp/master?style=flat-square)

A deepfake detection project based on a WGAN-GP training pipeline, with a web frontend that supports image upload inference and a separate report page.

## Repository

- GitHub: https://github.com/1234620/deepfake-detector-wgan-gp
- Branch: `master`

## Project Structure

- `model/` training code, model definitions, and configuration
- `checkpoints/` trained model checkpoints and threshold file
- `results/` evaluation artifacts (confusion matrix, ROC curve, loss curves)
- `backend/` Flask API server for model inference
- `frontend/` homepage (inference UI) and separate report page

## Quick Start

```bash
cd "/Users/ahmedmoosani/Downloads/1st Model"
python3 -m pip install -r model/requirements.txt
python3 -m pip install flask
```

Start backend + frontend:

```bash
bash frontend/start_frontend.sh
```

Open:

- Home page: `http://127.0.0.1:5500/`
- Report page: `http://127.0.0.1:5500/report`

If port 5500 is busy:

```bash
PID=$(lsof -ti :5500 || true); [ -n "$PID" ] && kill $PID
bash frontend/start_frontend.sh
```

## API

Health endpoint:

```bash
curl http://127.0.0.1:5500/api/health
```

Prediction endpoint:

```bash
curl -X POST http://127.0.0.1:5500/api/predict -F "image=@/path/to/image.jpg"
```

## Notes

- Inference uses `checkpoints/critic_final.pth` and `checkpoints/threshold.txt`.
- The frontend and backend are served by Flask from the project root.
- For training details, see `model/main.py` and `model/models.py`.
