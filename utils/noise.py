import torch
import numpy as np

class NoiseAdder:
    def __init__(self, noise_type='salt_and_pepper', noise_ratio=0.02):
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio

    # Adds noise to the images based on the specified noise type
    def add_noise(self, images):
        if self.noise_type == 'salt_and_pepper':
            return self._add_salt_pepper_noise(images)
        elif self.noise_type == 'gaussian':
            return self._add_gaussian_noise(images)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    # Adds salt and pepper noise to the images
    def _add_salt_pepper_noise(self, images):
        # Clone the original images to keep them unchanged
        noisy_images = images.clone()
        
        # Generate random number and coordinates for 'salt' pixels
        num_salt = int(self.noise_ratio * images.numel())
        coords = [np.random.randint(0, i, num_salt) for i in images.shape]
        noisy_images[coords[0], coords[1], coords[2], coords[3]] = 1

        # Generate random number and coordinates for 'pepper' pixels
        num_pepper = int(self.noise_ratio * images.numel())
        coords = [np.random.randint(0, i, num_pepper) for i in images.shape]
        noisy_images[coords[0], coords[1], coords[2], coords[3]] = 0

        return noisy_images

    # Adds Gaussian noise to the images
    def _add_gaussian_noise(self, images):
        mean = 0                    # Mean of the Gaussian noise
        std = self.noise_ratio      # Standard deviation of the Gaussian noise
        gaussian_noise = torch.randn(images.size()) * std + mean    # Generate Gaussian noise
        noisy_images = images + gaussian_noise                      # Add Gaussian noise to the images
        return noisy_images
