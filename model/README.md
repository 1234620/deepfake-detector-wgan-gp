# 🕵️ Deepfake Detection using WGAN-GP

> A deep learning project that trains an AI to tell the difference between **real human faces** and **deepfake (AI-generated) faces** — using a technique called **Wasserstein GAN with Gradient Penalty (WGAN-GP)**.

---

## 📖 Table of Contents
1. [What is this project?](#-what-is-this-project)
2. [How does it work? (Simple explanation)](#-how-does-it-work-simple-explanation)
3. [Project Structure](#-project-structure)
4. [The 8-Step Pipeline](#-the-8-step-pipeline)
5. [Setup & Installation](#-setup--installation)
6. [Preparing the Dataset](#-preparing-the-dataset)
7. [Running the Project](#-running-the-project)
8. [Output & Results](#-output--results)
9. [Configuration (Changing Settings)](#-configuration-changing-settings)
10. [Glossary (For Beginners)](#-glossary-for-beginners)

---

## 🤔 What is this project?

**Deepfakes** are fake videos or images of people that are created by AI — they look real but aren't.  
This project builds a system that can **automatically detect** whether a face image is real or a deepfake.

### The core idea:
Instead of just training a simple classifier, we train **two AI models that compete against each other**:

| Model | Role |
|-------|------|
| **Generator** | Tries to create convincing fake images from random noise |
| **Critic** | Tries to score how "real" an image is |

After training, we **throw away the Generator** and use only the **Critic as our deepfake detector**. The Critic has become so good at judging images that it can reliably detect deepfakes.

---

## 🧠 How does it work? (Simple explanation)

Imagine training a **counterfeit money detector**:

1. You show the detector **real money** (real faces) and **fake money** (deepfakes + AI-generated fakes)
2. A separate **counterfeiter** (Generator) keeps trying to make better fakes to fool the detector
3. Both get better over time through competition
4. After training — the **detector (Critic) is now an expert**
5. You fire the counterfeiter and keep only the expert detector

This competition is called a **GAN** (Generative Adversarial Network). The "W" and "GP" parts make the training more **stable and reliable** compared to older methods.

---

## 📁 Project Structure

```
ATML PROJECT/
│
├── config.py        ← All settings and hyperparameters (change things here)
├── models.py        ← Neural network architectures + loss functions
├── main.py          ← Full training + evaluation pipeline
├── README.md        ← This file
│
├── data/            ← YOU must create this with your dataset (see below)
│   ├── real/        ← Put real face images here
│   └── fake/        ← Put deepfake images here
│
├── checkpoints/     ← Auto-created: saved model weights during training
├── samples/         ← Auto-created: generated image previews per epoch
└── results/         ← Auto-created: evaluation plots (ROC curve, confusion matrix, etc.)
```

> **Note:** The `data/`, `checkpoints/`, `samples/`, and `results/` folders are created automatically when you run the project — except `data/` which you must fill with your dataset.

---

## 🔢 The 8-Step Pipeline

Here is exactly what happens when you run this project, from start to finish:

---

### STEP 1 — Dataset Preparation (`main.py`)
- Loads all images from `data/real/` and `data/fake/`
- Resizes every image to **64×64 pixels**
- Normalizes pixel values from [0, 255] → **[-1, 1]** (required for the neural network)
- Splits data into:
  - **70% Training** — the AI learns from this
  - **15% Validation** — used to tune the detection threshold
  - **15% Test** — final unseen evaluation

---

### STEP 2 — Generator Input (`models.py`)
- Creates a **random noise vector** of size 100 (called `z`)
- This random noise is the "seed" from which the Generator creates fake images
- Think of it like giving a random DNA code to create a unique fake face

---

### STEP 3 — Generator Architecture (`models.py`)
- The Generator takes the noise vector and **upsamples** it through 5 layers into a full image
- Each layer doubles the spatial size: `1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64`
- Uses **ConvTranspose2d** (reverse convolution), **BatchNorm**, and **ReLU** activations
- Final output: a fake face image of size **3×64×64** (RGB, 64×64 pixels)

```
Noise (100×1×1)
    ↓ ConvTranspose2d + BatchNorm + ReLU   → 512 channels, 4×4
    ↓ ConvTranspose2d + BatchNorm + ReLU   → 256 channels, 8×8
    ↓ ConvTranspose2d + BatchNorm + ReLU   → 128 channels, 16×16
    ↓ ConvTranspose2d + BatchNorm + ReLU   → 64 channels, 32×32
    ↓ ConvTranspose2d + Tanh               → 3 channels, 64×64
    = Fake Image
```

---

### STEP 4 — Critic Architecture (`models.py`)
- The Critic takes any image and outputs a **single real number (score)**
- **Higher score → more real** | **Lower score → more fake**
- ⚠️ Important differences from a normal classifier:
  - **No Sigmoid** (output is unbounded, not 0–1)
  - **No BatchNorm** (required for the gradient penalty to work correctly)
  - Uses **LeakyReLU** instead of ReLU

```
Image (3×64×64)
    ↓ Conv2d + LeakyReLU   → 64 channels, 32×32
    ↓ Conv2d + LeakyReLU   → 128 channels, 16×16
    ↓ Conv2d + LeakyReLU   → 256 channels, 8×8
    ↓ Conv2d + LeakyReLU   → 512 channels, 4×4
    ↓ Conv2d               → 1 value (Wasserstein score)
    = Realness Score
```

---

### STEP 5 — Loss Functions (`models.py`)
This is what the AI optimises (minimises). Unlike regular GANs, **no Binary Cross-Entropy (BCE)** is used.

**Critic Loss:**
$$\text{Loss}_{Critic} = \mathbb{E}[\text{fake score}] - \mathbb{E}[\text{real score}] + \lambda \times \text{Gradient Penalty}$$

The Critic wants to give **low scores to fakes** and **high scores to reals**.

**Generator Loss:**
$$\text{Loss}_{Generator} = -\mathbb{E}[\text{fake score}]$$

The Generator wants the Critic to give **high scores to its fakes**.

**Gradient Penalty** enforces a mathematical constraint (Lipschitz condition) that keeps training stable. λ = 10.

---

### STEP 6 — Training Loop (`main.py`)
For every batch of images, this repeats:

```
Repeat 5 times:
  1. Get real images from dataset
  2. Get deepfake images from dataset
  3. Generate fake images using Generator + random noise
  4. Compute Critic loss (on all 3 types)
  5. Compute gradient penalty
  6. Update Critic weights

Then once:
  7. Generate new fake images
  8. Compute Generator loss
  9. Update Generator weights

→ Repeat for 100 epochs
```

> Why train Critic 5× more? Because the Critic needs to properly approximate the Wasserstein distance before the Generator can meaningfully improve.

---

### STEP 7 — After Training: Critic becomes the Detector (`main.py`)
- The **Generator is discarded** — we no longer need it
- The trained **Critic is our deepfake detector**
- We find the best **threshold** score on the validation set:
  - `score > threshold` → **REAL**
  - `score < threshold` → **DEEPFAKE**

---

### STEP 8 — Evaluation (`main.py`)
Tests the detector on the unseen **test set** and reports:

| Metric | What it means |
|--------|--------------|
| **Accuracy** | % of all images classified correctly |
| **Precision** | Of images predicted Real, how many actually were? |
| **Recall** | Of all actual Real images, how many did we catch? |
| **F1-Score** | Balance between Precision and Recall |
| **Confusion Matrix** | Table showing all correct/incorrect predictions |
| **ROC Curve + AUC** | How well the model separates real from fake at all thresholds |

Plots are saved to the `results/` folder.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher ([download here](https://www.python.org/downloads/))
- A Kaggle account (for downloading the dataset)

### 1. Install required libraries
Open a terminal in the project folder and run:

```bash
pip install torch torchvision pillow matplotlib scikit-learn numpy
```

> If you have an NVIDIA GPU, install the GPU version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for much faster training.

---

## 🗂️ Preparing the Dataset

You need a dataset of **real face images** and **deepfake face images**.

**Recommended dataset:** [FaceForensics++ on Kaggle](https://www.kaggle.com/) or any deepfake detection dataset.

Once downloaded and extracted, organise your images like this:

```
ATML PROJECT/
└── data/
    ├── real/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    └── fake/
        ├── fake001.jpg
        ├── fake002.jpg
        └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

> The more images the better. Aim for at least **1000+ images per class** for reasonable results.

---

## ▶️ Running the Project

Once your dataset is in place, simply run:

```bash
python main.py
```

That's it! The script will automatically:
1. Load and preprocess your dataset
2. Train the WGAN-GP (this takes time — use a GPU if possible)
3. Find the best detection threshold
4. Evaluate and save all results

### Expected console output:
```
============================================================
  DEEPFAKE DETECTION USING WGAN-GP
============================================================
Device: cuda | Image: 64x64
Epochs: 100 | Batch: 64
============================================================

>>> STEP 1: Loading dataset...
[Dataset] Real: 70001 | Fake: 70001 | Total: 140002
[Split] Train: 98001 | Val: 21000 | Test: 21001

>>> STEPS 2-6: Training WGAN-GP...
Epoch [1/100] | Critic: -12.3456 | Generator: -0.8901
...
Epoch [100/100] | Critic: -2.1234 | Generator: -1.5678

>>> STEP 7: Finding detection threshold...
[Detector] Best threshold: 0.1234 | Val accuracy: 0.8900

>>> STEP 8: Evaluating on test set...
Accuracy  : 0.8823 (88.23%)
Precision : 0.8910
Recall    : 0.8750
F1-Score  : 0.8829
```

> **Note on Critic loss:** The displayed Critic loss is the **average across all 5 critic iterations per batch** (not just the last one). This gives a true representation of how well the Wasserstein distance is being approximated.

---

## 📊 Output & Results

After training completes, you'll find:

| File | Location | What it is |
|------|----------|------------|
| `critic_final.pth` | `checkpoints/` | Trained Critic model weights (your detector) |
| `generator_final.pth` | `checkpoints/` | Trained Generator weights |
| `threshold.txt` | `checkpoints/` | The best detection threshold value |
| `epoch_10.png`, `epoch_20.png`... | `samples/` | Images generated by the Generator each epoch |
| `confusion_matrix.png` | `results/` | Visual breakdown of correct/wrong predictions |
| `roc_curve.png` | `results/` | ROC curve with AUC score |
| `training_losses.png` | `results/` | Critic & Generator loss over all epochs |

---

## 🔧 Configuration (Changing Settings)

All settings are in [config.py](config.py). You can safely change these without touching any other file:

```python
IMAGE_SIZE = 64          # Try 128 if you have a strong GPU
NUM_EPOCHS = 100         # More epochs = better model (but slower)
BATCH_SIZE = 64          # Reduce to 32 if you run out of GPU memory
LEARNING_RATE = 1e-4     # Don't change unless you know what you're doing
CRITIC_ITERATIONS = 5    # Standard for WGAN-GP
LAMBDA_GP = 10           # Gradient penalty weight (standard value)

REAL_DIR = "data/real"   # Path to your real images
FAKE_DIR = "data/fake"   # Path to your deepfake images
```

---

## 📚 Glossary (For Beginners)

| Term | Simple meaning |
|------|---------------|
| **GAN** | Two AIs competing — one creates fakes, one detects them |
| **WGAN-GP** | A more stable version of GAN using Wasserstein distance + Gradient Penalty |
| **Generator** | The AI that creates fake images from random noise |
| **Critic** | The AI that scores how real an image looks (replaces "Discriminator") |
| **Discriminator** | Old name for Critic in regular GANs — outputs 0 or 1 |
| **Wasserstein distance** | A better mathematical measure of how different two distributions are |
| **Gradient Penalty** | A technique to keep training stable by constraining how fast the Critic changes |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | A small group of images processed together (64 at a time here) |
| **Loss** | A number measuring how wrong the model is — lower = better |
| **Threshold** | The score cutoff: above it = Real, below it = Deepfake |
| **ROC Curve** | A graph showing detector performance at every possible threshold |
| **AUC** | Area Under the ROC Curve — closer to 1.0 = better detector |
| **Confusion Matrix** | A table showing True Positives, False Positives, True Negatives, False Negatives |
| **Precision** | How trustworthy "Real" predictions are |
| **Recall** | How many actual real images were correctly found |
| **F1-Score** | Single number balancing Precision and Recall |
| **BatchNorm** | A technique to stabilise training (NOT used in Critic for WGAN-GP) |
| **LeakyReLU** | An activation function that allows small negative values (better for Critic) |
| **Tanh** | An activation function that squishes output to [-1, 1] |
| **ConvTranspose2d** | "Reverse convolution" — used to upsample/grow an image |
| **Checkpoint** | A saved snapshot of model weights during training |

---

## 👤 Author Notes

- This project is built entirely in **PyTorch**
- Training on **CPU is possible but slow** — a GPU (NVIDIA) is strongly recommended
- The project is **fully reproducible** — random seed is fixed at 42 in all libraries
- The `data/` folder is the only thing you need to provide — everything else is automated

---
