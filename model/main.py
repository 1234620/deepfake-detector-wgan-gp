# main.py — Complete Deepfake Detection Pipeline using WGAN-GP
# ============================================================
# Runs all 8 steps end-to-end:
#   STEP 1: Dataset preparation (load, resize, normalize, split)
#   STEP 2: Generator input (noise vector z)
#   STEP 3: Generator creates fake images
#   STEP 4: Critic evaluates real, deepfake, and generated images
#   STEP 5: WGAN-GP losses (Wasserstein + gradient penalty)
#   STEP 6: Training loop (Critic 5×, Generator 1×)
#   STEP 7: Discard Generator, use Critic as detector + threshold
#   STEP 8: Evaluate (Accuracy, Precision, Recall, F1, CM, ROC)
#
# Usage:  python main.py
# ============================================================

import os                                       # File system operations
import random                                   # Python random seed
import numpy as np                              # Numerical operations
import torch                                    # Core PyTorch
import torch.optim as optim                     # Optimizers
from torch.utils.data import (                  # Data utilities
    Dataset, DataLoader, random_split,
)
from torchvision import transforms              # Image transforms
from torchvision.utils import save_image        # Save image grids
from PIL import Image                           # Load image files
import matplotlib                               # Plotting backend
matplotlib.use("Agg")                           # Non-interactive backend
import matplotlib.pyplot as plt                 # Plotting
from sklearn.metrics import (                   # Evaluation metrics
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

import config                                   # All hyperparameters
from models import (                            # Models + losses
    Generator, Critic, generate_noise,
    compute_gradient_penalty, compute_critic_loss, compute_generator_loss,
)


# STEP 1 — DATASET PREPARATION
# ====================================================================
# Load real + deepfake images, resize to 64×64, normalize to [-1,1],
# split into Train (70%) / Validation (15%) / Test (15%)

class DeepfakeDataset(Dataset):
    """Loads real images (label=1) and deepfake images (label=0)."""

    def __init__(self, real_dir, fake_dir, transform=None):
        super().__init__()
        self.transform = transform                              # Image preprocessing pipeline
        self.image_paths = []                                   # All image file paths
        self.labels = []                                        # 1=real, 0=fake

        # Collect REAL images → label 1
        for f in os.listdir(real_dir):                          # Loop through real folder
            path = os.path.join(real_dir, f)                    # Full file path
            if self._is_image(path):                            # Only image files
                self.image_paths.append(path)                   # Store path
                self.labels.append(1)                           # Real = 1

        # Collect FAKE (deepfake) images → label 0
        for f in os.listdir(fake_dir):                          # Loop through fake folder
            path = os.path.join(fake_dir, f)                    # Full file path
            if self._is_image(path):                            # Only image files
                self.image_paths.append(path)                   # Store path
                self.labels.append(0)                           # Fake = 0

        print(f"[Dataset] Real: {self.labels.count(1)} | Fake: {self.labels.count(0)} | Total: {len(self.labels)}")

    @staticmethod
    def _is_image(path):
        """Check if file is a supported image format."""
        return path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))

    def __len__(self):
        return len(self.image_paths)                            # Total number of images

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Load as RGB
        if self.transform:
            image = self.transform(image)                       # Apply resize + normalize
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Label as tensor
        return image, label


def get_dataloaders():
    """Create Train/Val/Test DataLoaders with preprocessing."""

    # Image transform: Resize → Tensor → Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # Resize to 64×64
        transforms.ToTensor(),                                       # Convert to tensor [0,1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),                 # Normalize to [-1,1]
    ])

    # Load full dataset
    dataset = DeepfakeDataset(config.REAL_DIR, config.FAKE_DIR, transform)

    # Split: 70% train, 15% val, 15% test
    total = len(dataset)
    train_n = int(total * config.TRAIN_RATIO)                   # 70%
    val_n = int(total * config.VAL_RATIO)                       # 15%
    test_n = total - train_n - val_n                            # Remaining → test
    gen = torch.Generator().manual_seed(config.SEED)            # Reproducible split
    train_set, val_set, test_set = random_split(dataset, [train_n, val_n, test_n], generator=gen)
    print(f"[Split] Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# STEP 6 — TRAINING LOOP
# ====================================================================
# Per batch: train Critic 5× (on real + deepfake + generated),
# then train Generator 1×.  Both deepfake + generated = fake.

def train_wgan_gp(train_loader):
    """Train WGAN-GP and return trained Generator, Critic, and loss histories."""

    os.makedirs(config.SAVE_DIR, exist_ok=True)                # Create checkpoint folder
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)              # Create sample folder

    # Instantiate models
    gen = Generator().to(config.DEVICE)                        # Generator on device
    crit = Critic().to(config.DEVICE)                          # Critic on device

    # Adam optimizers with WGAN-GP recommended betas (0.0, 0.9)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    opt_crit = optim.Adam(crit.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    fixed_noise = generate_noise(64)                           # Fixed noise to track generator progress
    gen_losses, crit_losses = [], []                           # Loss histories

    print("=" * 60)
    print(f"Training WGAN-GP | Device: {config.DEVICE} | Epochs: {config.NUM_EPOCHS}")
    print(f"Critic iters per gen update: {config.CRITIC_ITERATIONS} | lambda_GP: {config.LAMBDA_GP}")
    print("=" * 60)

    for epoch in range(config.NUM_EPOCHS):                     # Loop over all epochs
        epoch_c, epoch_g, n_batches = 0.0, 0.0, 0
        gen.train()                                            # Generator → training mode
        crit.train()                                           # Critic → training mode

        for images, labels in train_loader:                    # Iterate batches
            images = images.to(config.DEVICE)                  # Move images to device
            labels = labels.to(config.DEVICE)                  # Move labels to device

            # Separate real (label=1) and deepfake (label=0) images
            real_imgs = images[labels == 1]                    # Real images from batch
            deep_imgs = images[labels == 0]                    # Deepfake images from batch

            if real_imgs.size(0) == 0 or deep_imgs.size(0) == 0:
                continue                                       # Skip if batch lacks one class

            # Balance to equal counts
            k = min(real_imgs.size(0), deep_imgs.size(0))      # Smaller count
            real_imgs = real_imgs[:k]                          # Trim real
            deep_imgs = deep_imgs[:k]                          # Trim fake

            # === CRITIC TRAINING (5 iterations) ===
            for _ in range(config.CRITIC_ITERATIONS):          # Train critic 5 times
                noise = generate_noise(k)                      # Random noise
                gen_fakes = gen(noise).detach()                # Generator fakes (detached)

                # Critic scores all 3 image types
                sc_real = crit(real_imgs)                       # Scores for REAL (want HIGH)
                sc_deep = crit(deep_imgs)                       # Scores for DEEPFAKE dataset (want LOW)
                sc_gfake = crit(gen_fakes)                      # Scores for GENERATOR fakes (want LOW)

                # Combine deepfake + generator fakes (both treated as FAKE)
                all_fakes = torch.cat([deep_imgs, gen_fakes], dim=0)
                sc_all_fake = torch.cat([sc_deep, sc_gfake], dim=0)

                # Gradient penalty (interpolate between real and all fakes)
                real_gp = real_imgs.repeat(2, 1, 1, 1)[:all_fakes.size(0)]  # Match fake count
                gp = compute_gradient_penalty(crit, real_gp, all_fakes)

                # Critic loss = E[fake] - E[real] + lambda*GP
                c_loss = compute_critic_loss(sc_real, sc_all_fake, gp)

                opt_crit.zero_grad()                           # Clear gradients
                c_loss.backward()                              # Backpropagate
                opt_crit.step()                                # Update critic weights
                epoch_c += c_loss.item()                       # Accumulate critic loss (all 5 iters)

            # === GENERATOR TRAINING (1 iteration) ===
            noise = generate_noise(k)                          # Fresh noise
            gen_fakes = gen(noise)                              # Generate fakes (graph attached)
            sc_gen = crit(gen_fakes)                            # Critic scores the fakes

            g_loss = compute_generator_loss(sc_gen)            # Gen loss = -E[fake_score]

            opt_gen.zero_grad()                                # Clear gradients
            g_loss.backward()                                  # Backpropagate
            opt_gen.step()                                     # Update generator weights

            epoch_g += g_loss.item()                           # Accumulate generator loss
            n_batches += 1                                     # Count batches

        # Epoch summary
        avg_c = epoch_c / max(n_batches * config.CRITIC_ITERATIONS, 1)  # Average critic loss
        avg_g = epoch_g / max(n_batches, 1)                    # Average generator loss
        crit_losses.append(avg_c)                              # Store for plotting
        gen_losses.append(avg_g)                               # Store for plotting
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] | Critic: {avg_c:.4f} | Generator: {avg_g:.4f}")

        # Save samples and checkpoints periodically
        if (epoch + 1) % config.SAVE_EVERY == 0:
            gen.eval()
            with torch.no_grad():
                samples = gen(fixed_noise)                     # Generate from fixed noise
                save_image(samples, os.path.join(config.SAMPLE_DIR, f"epoch_{epoch+1}.png"),
                           normalize=True, nrow=8)
            gen.train()
            torch.save(gen.state_dict(), os.path.join(config.SAVE_DIR, f"gen_epoch_{epoch+1}.pth"))
            torch.save(crit.state_dict(), os.path.join(config.SAVE_DIR, f"crit_epoch_{epoch+1}.pth"))
            print(f"  -> Saved checkpoint & samples at epoch {epoch+1}")

    # Save final models
    torch.save(gen.state_dict(), os.path.join(config.SAVE_DIR, "generator_final.pth"))
    torch.save(crit.state_dict(), os.path.join(config.SAVE_DIR, "critic_final.pth"))
    print("Training complete! Final models saved.")

    return gen, crit, gen_losses, crit_losses


# STEP 7 — AFTER TRAINING: Discard Generator, Use Critic as Detector
# ====================================================================
# Find threshold on validation set: score > threshold → Real
#                                    score < threshold → Deepfake

def find_best_threshold(critic, val_loader):
    """Search for the threshold that maximises accuracy on the validation set."""
    all_scores, all_labels = [], []

    critic.eval()                                              # Evaluation mode
    with torch.no_grad():                                      # No gradients needed
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            scores = critic(images).reshape(-1).cpu().numpy()  # reshape(-1) safe for any batch size
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Try 1000 thresholds between min and max score
    best_acc, best_thresh = 0.0, 0.0
    for t in np.linspace(all_scores.min(), all_scores.max(), 1000):
        preds = (all_scores > t).astype(int)                   # Score > t → Real (1)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    print(f"[Detector] Best threshold: {best_thresh:.4f} | Val accuracy: {best_acc:.4f}")
    return best_thresh


def detect_single_image(critic, image_path, threshold):
    """Classify a single image as REAL or DEEPFAKE using the trained Critic."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    image = Image.open(image_path).convert("RGB")              # Load image
    image = transform(image).unsqueeze(0).to(config.DEVICE)    # Preprocess + batch dim

    with torch.no_grad():
        score = critic(image).item()                           # Get Wasserstein score

    prediction = "REAL" if score > threshold else "DEEPFAKE"   # Apply threshold
    print(f"Image: {image_path} | Score: {score:.4f} | Threshold: {threshold:.4f} | -> {prediction}")
    return prediction, score


# STEP 8 — EVALUATION (Accuracy, Precision, Recall, F1, CM, ROC)

def evaluate(critic, test_loader, threshold):
    """Evaluate the detector on the unseen test set and save plots."""

    print("\n" + "=" * 60)
    print("STEP 8: EVALUATION ON TEST SET")
    print("=" * 60)

    # Collect predictions for all test images
    all_preds, all_scores, all_labels = [], [], []
    critic.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            scores = critic(images).reshape(-1).cpu().numpy()  # reshape(-1) safe for any batch size
            preds = (scores > threshold).astype(int)           # Apply threshold
            all_preds.extend(preds)
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)                 # Accuracy
    prec = precision_score(all_labels, all_preds, zero_division=0)  # Precision
    rec = recall_score(all_labels, all_preds, zero_division=0)      # Recall
    f1 = f1_score(all_labels, all_preds, zero_division=0)           # F1-Score
    cm = confusion_matrix(all_labels, all_preds)                    # Confusion Matrix

    # Print results
    print(f"\nThreshold: {threshold:.4f} | Samples: {len(all_labels)}")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"             Pred Fake  Pred Real")
    print(f"  True Fake: {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"  True Real: {cm[1][0]:>8}  {cm[1][1]:>8}")

    # Save plots
    os.makedirs("results", exist_ok=True)

    # --- Confusion Matrix plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Fake (0)", "Real (1)"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - Deepfake Detection")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()

    # --- ROC Curve plot ---
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Deepfake Detection"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150)
    plt.close()

    print(f"\n[Saved] confusion_matrix.png, roc_curve.png -> results/")
    print(f"[AUC] {roc_auc:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": roc_auc, "cm": cm}


# MAIN — Run the complete pipeline

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(config.SEED)                                      # Fix seeds (42)

    print("=" * 60)
    print("  DEEPFAKE DETECTION USING WGAN-GP")
    print("=" * 60)
    print(f"Device: {config.DEVICE} | Image: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS} | Batch: {config.BATCH_SIZE}")
    print("=" * 60)

    # STEP 1: Load dataset
    print("\n>>> STEP 1: Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # STEPS 2-6: Train WGAN-GP
    print("\n>>> STEPS 2-6: Training WGAN-GP...")
    gen, critic, gen_losses, crit_losses = train_wgan_gp(train_loader)

    # STEP 7: Find threshold (discard Generator, keep Critic)
    print("\n>>> STEP 7: Finding detection threshold...")
    critic.eval()
    threshold = find_best_threshold(critic, val_loader)

    # STEP 8: Evaluate on test set
    print("\n>>> STEP 8: Evaluating on test set...")
    results = evaluate(critic, test_loader, threshold)

    # Save threshold
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    with open(os.path.join(config.SAVE_DIR, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    # Plot training loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gen_losses, label="Generator Loss")
    ax.plot(crit_losses, label="Critic Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("WGAN-GP Training Loss Curves"); ax.legend()
    plt.tight_layout()
    plt.savefig("results/training_losses.png", dpi=150)
    plt.close()
    print(f"[Saved] training_losses.png -> results/")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Critic saved: {os.path.join(config.SAVE_DIR, 'critic_final.pth')}")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
