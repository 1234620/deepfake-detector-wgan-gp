# config.py — All hyperparameters and settings in one place
# Change any setting here without touching model or training code.
# =================================================================

import torch  # Import PyTorch to check for GPU availability

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto GPU/CPU

# Image Settings
IMAGE_SIZE = 64          # Resize all images to 64x64 pixels
IMAGE_CHANNELS = 3       # 3 channels for RGB colour images

# Generator Settings
NOISE_DIM = 100          # Dimension of random noise vector z (Generator input)
GEN_FEATURE_MAP = 64     # Base number of feature maps in Generator

# Critic Settings
CRITIC_FEATURE_MAP = 64  # Base number of feature maps in Critic

# Training Hyperparameters
BATCH_SIZE = 64          # Images per forward/backward pass
NUM_EPOCHS = 100         # Total training epochs (50–200 recommended)
LEARNING_RATE = 1e-4     # Adam optimizer step size (0.0001)
BETA1 = 0.0              # Adam beta1 (0.0 recommended for WGAN-GP)
BETA2 = 0.9              # Adam beta2 (0.9 recommended for WGAN-GP)
CRITIC_ITERATIONS = 5    # Train Critic 5 times per 1 Generator update
LAMBDA_GP = 10           # Gradient penalty coefficient (Lipschitz constraint)

# Dataset Paths 
REAL_DIR = "data/real"   # Folder containing real face images
FAKE_DIR = "data/fake"   # Folder containing deepfake face images

# Train / Validation / Test Split Ratios
TRAIN_RATIO = 0.7        # 70% training
VAL_RATIO = 0.15         # 15% validation (threshold tuning)
TEST_RATIO = 0.15        # 15% testing (final evaluation)

# Saving
SAVE_DIR = "checkpoints" # Folder for model weights
SAMPLE_DIR = "samples"   # Folder for generated image samples
SAVE_EVERY = 10          # Save checkpoint every N epochs

# Reproducibility
SEED = 42                # Random seed for reproducible results
