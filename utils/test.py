import tensorflow as tf
from utils import plot_images

def test_model(generator, noise_adder, test_data):
    for i, (images, _) in enumerate(test_data):
        noisy_images = noise_adder.add_noise(images)

        # Ensure images have 4 dimensions
        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=-1)
        if len(noisy_images.shape) == 3:
            noisy_images = tf.expand_dims(noisy_images, axis=-1)

        reconstructed_images = generator(noisy_images, training=False)

        # Visualize or save the images
        plot_images(images, noisy_images, reconstructed_images, num_images=min(5, images.shape[0]), show=True, save=True, epoch=0, batch=i, model_seed=None)
