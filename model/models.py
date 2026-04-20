# ============================================================
# models.py — Generator + Critic + Loss Functions
# ============================================================
# Contains:
#   STEP 2: Generator input (noise vector z, dim=100)
#   STEP 3: Generator architecture (DCGAN-style)
#   STEP 4: Critic architecture (NO Sigmoid, NO BatchNorm)
#   STEP 5: WGAN-GP loss functions + gradient penalty
# ============================================================

import torch                            # Core PyTorch library
import torch.nn as nn                   # Neural network modules
import torch.autograd as autograd       # For manual gradient computation (gradient penalty)

import config                           # Centralised settings


# STEPS 2 & 3 — GENERATOR (DCGAN-style)
# ====================================================================
# Takes random noise z (100×1×1) → upsamples → fake image (3×64×64)
# Uses: ConvTranspose2d + BatchNorm + ReLU, final layer uses Tanh

class Generator(nn.Module):
    """Transforms noise vector z into a fake image (3×64×64)."""

    def __init__(self):
        super(Generator, self).__init__()                       # Initialise parent class

        ngf = config.GEN_FEATURE_MAP                            # Base feature maps (64)
        nz = config.NOISE_DIM                                   # Noise dimension (100)
        nc = config.IMAGE_CHANNELS                              # Output channels (3 = RGB)

        self.network = nn.Sequential(
            # Block 1: (batch, 100, 1, 1) → (batch, 512, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),     # Upsample noise to 4×4
            nn.BatchNorm2d(ngf * 8),                                    # BatchNorm over 512 channels
            nn.ReLU(inplace=True),                                      # ReLU activation

            # Block 2: (batch, 512, 4, 4) → (batch, 256, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # Stride 2 doubles: 4→8
            nn.BatchNorm2d(ngf * 4),                                     # BatchNorm over 256 channels
            nn.ReLU(inplace=True),                                       # ReLU activation

            # Block 3: (batch, 256, 8, 8) → (batch, 128, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # Stride 2 doubles: 8→16
            nn.BatchNorm2d(ngf * 2),                                     # BatchNorm over 128 channels
            nn.ReLU(inplace=True),                                       # ReLU activation

            # Block 4: (batch, 128, 16, 16) → (batch, 64, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # Stride 2 doubles: 16→32
            nn.BatchNorm2d(ngf),                                         # BatchNorm over 64 channels
            nn.ReLU(inplace=True),                                       # ReLU activation

            # Block 5 (Final): (batch, 64, 32, 32) → (batch, 3, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # Stride 2 doubles: 32→64
            nn.Tanh(),                                                    # Tanh → output in [-1, 1]
        )

        self._initialize_weights()                              # Apply DCGAN weight init

    def _initialize_weights(self):
        """DCGAN weight init: ConvTranspose2d ~ N(0, 0.02), BatchNorm weight ~ N(1, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)           # Weights ~ N(0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)           # Weight ~ N(1, 0.02)
                nn.init.constant_(m.bias, 0)                    # Bias = 0

    def forward(self, z):
        """Forward: noise z (batch, 100, 1, 1) → fake image (batch, 3, 64, 64)."""
        return self.network(z)


# STEP 4 — CRITIC (Wasserstein Critic, NOT a Discriminator)
# ====================================================================
# Takes an image (3×64×64) → outputs a real-valued Wasserstein score
# NO Sigmoid (unbounded output), NO BatchNorm (required for GP)
# Higher score = more real, Lower score = more fake

class Critic(nn.Module):
    """Assigns a Wasserstein realness score to an image. No Sigmoid, No BatchNorm."""

    def __init__(self):
        super(Critic, self).__init__()                          # Initialise parent class

        ndf = config.CRITIC_FEATURE_MAP                         # Base feature maps (64)
        nc = config.IMAGE_CHANNELS                              # Input channels (3 = RGB)

        self.network = nn.Sequential(
            # Block 1: (batch, 3, 64, 64) → (batch, 64, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),           # Stride 2 halves: 64→32
            nn.LeakyReLU(0.2, inplace=True),                    # LeakyReLU (slope 0.2)
            # NO BatchNorm in Critic!

            # Block 2: (batch, 64, 32, 32) → (batch, 128, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),      # Stride 2 halves: 32→16
            nn.LeakyReLU(0.2, inplace=True),                    # LeakyReLU
            # NO BatchNorm!

            # Block 3: (batch, 128, 16, 16) → (batch, 256, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # Stride 2 halves: 16→8
            nn.LeakyReLU(0.2, inplace=True),                    # LeakyReLU
            # NO BatchNorm!

            # Block 4: (batch, 256, 8, 8) → (batch, 512, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # Stride 2 halves: 8→4
            nn.LeakyReLU(0.2, inplace=True),                    # LeakyReLU
            # NO BatchNorm!

            # Final: (batch, 512, 4, 4) → (batch, 1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),        # 4×4 kernel collapses spatial dims
            # NO Sigmoid! Output is unbounded Wasserstein score
        )

        self._initialize_weights()                              # Apply weight init

    def _initialize_weights(self):
        """Conv2d weight init: ~ N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)           # Weights ~ N(0, 0.02)

    def forward(self, image):
        """Forward: image (batch, 3, 64, 64) → score (batch, 1). Higher = more real."""
        output = self.network(image)                            # Pass through conv layers
        return output.view(output.size(0), -1)                  # Flatten to (batch, 1)


# STEP 2 HELPER — Generate random noise vectors

def generate_noise(batch_size):
    """Create random noise z of shape (batch_size, 100, 1, 1) on the correct device."""
    return torch.randn(batch_size, config.NOISE_DIM, 1, 1, device=config.DEVICE)


# STEP 5 — LOSS FUNCTIONS (WGAN-GP — NO BCE!)
# ====================================================================
# Critic Loss  = E[fake_score] - E[real_score] + λ * GradientPenalty
# Generator Loss = -E[fake_score]
# Gradient Penalty enforces Lipschitz constraint via interpolation

def compute_gradient_penalty(critic, real_images, fake_images):
    """
    Gradient Penalty: interpolate between real & fake, penalise if
    gradient norm deviates from 1.  GP = E[(||∇ critic(interp)|| - 1)²]
    """
    batch_size = real_images.size(0)                            # Current batch size

    # Random interpolation coefficient per image
    epsilon = torch.rand(batch_size, 1, 1, 1, device=config.DEVICE)  # Uniform [0,1]

    # Interpolated images = mix of real and fake
    interpolated = (epsilon * real_images + (1 - epsilon) * fake_images)  # Linear mix
    interpolated.requires_grad_(True)                           # Enable gradient tracking

    # Critic scores for interpolated images
    scores = critic(interpolated)                               # Forward pass

    # Compute gradients of scores w.r.t. interpolated images
    gradients = autograd.grad(
        outputs=scores,                                         # Differentiate this
        inputs=interpolated,                                    # With respect to this
        grad_outputs=torch.ones_like(scores),                   # Upstream gradient = 1
        create_graph=True,                                      # Keep graph for 2nd order grads
        retain_graph=True,                                      # Don't free graph
    )[0]                                                        # Take first element of tuple

    gradients = gradients.view(batch_size, -1)                  # Flatten to (batch, 3*64*64)
    gradient_norm = gradients.norm(2, dim=1)                    # L2 norm per sample
    penalty = ((gradient_norm - 1) ** 2).mean()                 # Penalise deviation from 1

    return penalty                                              # Scalar GP value


def compute_critic_loss(score_real, score_all_fake, gradient_penalty):
    """Critic loss = E[fake_score] - E[real_score] + λ * GP."""
    return score_all_fake.mean() - score_real.mean() + config.LAMBDA_GP * gradient_penalty


def compute_generator_loss(score_fake):
    """Generator loss = -E[fake_score]. Generator wants HIGH scores for its fakes."""
    return -score_fake.mean()
