import matplotlib.pyplot as plt
import numpy as np

def plot_images(original, noisy, reconstructed, num_images=5):
    num_images = min(num_images, original.shape[0])

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        original_image = original[i].numpy().squeeze()
        noisy_image = noisy[i].numpy().squeeze()
        reconstructed_image = reconstructed[i].numpy().squeeze()

        plt.subplot(3, num_images, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(noisy_image, cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
        
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()