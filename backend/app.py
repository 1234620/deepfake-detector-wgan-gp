from __future__ import annotations

import io
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
MODEL_DIR = ROOT_DIR / "model"

# The training code expects absolute imports like `import config`.
sys.path.insert(0, str(MODEL_DIR))

import config  # type: ignore  # noqa: E402
from models import Critic  # type: ignore  # noqa: E402

app = Flask(__name__)

MODEL_STATE = {
    "ready": False,
    "error": "",
    "threshold": None,
    "device": str(config.DEVICE),
}

critic_model: Critic | None = None


def load_threshold() -> float:
    threshold_file = CHECKPOINTS_DIR / "threshold.txt"
    if not threshold_file.exists():
        raise FileNotFoundError(f"Missing threshold file: {threshold_file}")
    return float(threshold_file.read_text(encoding="utf-8").strip())


def load_critic_model() -> Critic:
    critic_path = CHECKPOINTS_DIR / "critic_final.pth"
    if not critic_path.exists():
        raise FileNotFoundError(f"Missing critic checkpoint: {critic_path}")

    model = Critic().to(config.DEVICE)
    state_dict = torch.load(critic_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def initialize_runtime() -> None:
    global critic_model

    try:
        critic_model = load_critic_model()
        threshold = load_threshold()
        MODEL_STATE["threshold"] = threshold
        MODEL_STATE["ready"] = True
        MODEL_STATE["error"] = ""
    except Exception as exc:  # pragma: no cover - startup fault path
        MODEL_STATE["ready"] = False
        MODEL_STATE["error"] = str(exc)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    return tensor


@app.route("/")
def home() -> object:
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/report")
def report_page() -> object:
    return send_from_directory(FRONTEND_DIR, "report.html")


@app.route("/frontend/<path:filename>")
def frontend_assets(filename: str) -> object:
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/results/<path:filename>")
def results_assets(filename: str) -> object:
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/checkpoints/<path:filename>")
def checkpoint_assets(filename: str) -> object:
    return send_from_directory(CHECKPOINTS_DIR, filename)


@app.get("/api/health")
def api_health() -> object:
    payload = {
        "ready": MODEL_STATE["ready"],
        "error": MODEL_STATE["error"],
        "device": MODEL_STATE["device"],
        "threshold": MODEL_STATE["threshold"],
        "artifacts": {
            "critic": (CHECKPOINTS_DIR / "critic_final.pth").exists(),
            "threshold": (CHECKPOINTS_DIR / "threshold.txt").exists(),
            "confusion_matrix": (RESULTS_DIR / "confusion_matrix.png").exists(),
            "roc_curve": (RESULTS_DIR / "roc_curve.png").exists(),
            "training_losses": (RESULTS_DIR / "training_losses.png").exists(),
        },
    }
    return jsonify(payload)


@app.post("/api/predict")
def api_predict() -> object:
    if not MODEL_STATE["ready"] or critic_model is None:
        return jsonify({"error": "Model is not ready", "details": MODEL_STATE["error"]}), 503

    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_bytes = image_file.read()
        input_tensor = preprocess_image(image_bytes)
    except Exception as exc:
        return jsonify({"error": "Invalid image input", "details": str(exc)}), 400

    with torch.no_grad():
        score = float(critic_model(input_tensor).reshape(-1)[0].item())

    threshold = float(MODEL_STATE["threshold"])
    prediction = "REAL" if score > threshold else "DEEPFAKE"

    return jsonify(
        {
            "prediction": prediction,
            "score": score,
            "threshold": threshold,
            "device": MODEL_STATE["device"],
        }
    )


initialize_runtime()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5500, debug=False)
