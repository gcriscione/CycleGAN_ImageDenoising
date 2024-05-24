import tensorflow as tf
import numpy as np

class NoiseAdder:
    def __init__(self, config):
        self.noise_type = config["noise_adder"]["noise_type"]
        self.salt_pepper_ratio = config["noise_adder"]["salt_pepper_ratio"]
        self.gaussian_mean = config["noise_adder"]["gaussian_mean"]
        self.gaussian_std = config["noise_adder"]["gaussian_std"]

    def add_noise(self, images):
        if self.noise_type == "salt_and_pepper":
            return self._add_salt_and_pepper_noise(images)
        elif self.noise_type == "gaussian":
            return self._add_gaussian_noise(images)
        else:
            raise ValueError("Unsupported noise type")

    def _add_salt_and_pepper_noise(self, images):
        images = images.numpy()  # Convert to numpy array
        noisy_images = images.copy()
        num_salt = np.ceil(self.salt_pepper_ratio * images.size * 0.5)
        num_pepper = np.ceil(self.salt_pepper_ratio * images.size * 0.5)

        # Add Salt noise
        coords = [np.random.randint(0, i, int(num_salt)) for i in images.shape]
        noisy_images[coords[0], coords[1], coords[2]] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i, int(num_pepper)) for i in images.shape]
        noisy_images[coords[0], coords[1], coords[2]] = 0

        return tf.convert_to_tensor(noisy_images)  # Convert back to tensor

    def _add_gaussian_noise(self, images):
        mean = self.gaussian_mean
        std = self.gaussian_std
        gaussian_noise = np.random.normal(mean, std, images.shape)
        noisy_images = images + gaussian_noise
        return tf.convert_to_tensor(np.clip(noisy_images, -1.0, 1.0))