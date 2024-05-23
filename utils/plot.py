import matplotlib.pyplot as plt
import numpy as np

def plot_images(original, noisy, denoised, n_images=5):
    original = original.cpu().detach().numpy()
    noisy = noisy.cpu().detach().numpy()
    denoised = denoised.cpu().detach().numpy()

    fig, axes = plt.subplots(nrows=n_images, ncols=3, figsize=(10, n_images * 3))
    for i in range(n_images):
        axes[i, 0].imshow(original[i, 0], cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy[i, 0], cmap='gray')
        axes[i, 1].set_title("Noisy")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(denoised[i, 0], cmap='gray')
        axes[i, 2].set_title("Denoised")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
