import tensorflow as tf
from models import Generator, Discriminator, Losses
from utils import MNISTDataLoader, NoiseAdder, plot_images, stats
from config.config_file import config
import json
import logging

CONFIGURATION_FILE = "result/setup/config.json"
MODEL_FILE = "result/setup/models.txt"

# Configure logger for training
training_logger = logging.getLogger('TrainingLogger')
training_logger.setLevel(logging.INFO)
handler = logging.FileHandler('result/logs/training.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
training_logger.addHandler(handler)

# Enable memory growth for GPUs
def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# Print configuration details
def print_config(obj):
    for key, value in obj.items():
        print(f"{key}: {value}")

# Initialize models, optimizers, and other components
def initialize_components(config):
    data_loader = MNISTDataLoader(config)
    train_loader = data_loader.get_train_data()
    G1 = Generator(config)
    G2 = Generator(config)
    D1 = Discriminator(config)
    D2 = Discriminator(config)

    optimizer_G = tf.keras.optimizers.Adam(
        learning_rate=config['general']['learning_rate'], 
        beta_1=config['general']['beta1'], 
        beta_2=config['general']['beta2']
    )
    optimizer_D1 = tf.keras.optimizers.Adam(
        learning_rate=config['general']['learning_rate'], 
        beta_1=config['general']['beta1'], 
        beta_2=config['general']['beta2']
    )
    optimizer_D2 = tf.keras.optimizers.Adam(
        learning_rate=config['general']['learning_rate'], 
        beta_1=config['general']['beta1'], 
        beta_2=config['general']['beta2']
    )

    noise_adder = NoiseAdder(config)
    losses = Losses(config)

    return train_loader, G1, G2, D1, D2, optimizer_G, optimizer_D1, optimizer_D2, noise_adder, losses

# Perform training for one epoch
def train_one_epoch(config, epoch, train_loader, G1, G2, D1, D2, optimizer_G, optimizer_D1, optimizer_D2, noise_adder, losses):
    for i, images in enumerate(train_loader):
        noisy_images = noise_adder.add_noise(images)

        # Ensure images have 4 dimensions
        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=-1)
        if len(noisy_images.shape) == 3:
            noisy_images = tf.expand_dims(noisy_images, axis=-1)

        with tf.GradientTape(persistent=True) as tape:
            fake_images_G2 = G2(noisy_images, training=True)
            fake_images_G1 = G1(images, training=True)

            real_output_D1 = D1(images, training=True)
            fake_output_D1 = D1(fake_images_G2, training=True)

            real_output_D2 = D2(noisy_images, training=True)
            fake_output_D2 = D2(fake_images_G1, training=True)

            loss_D1 = losses.discriminator_loss(real_output_D1, fake_output_D1)
            loss_D2 = losses.discriminator_loss(real_output_D2, fake_output_D2)

            loss_G1 = losses.generator_loss(fake_output_D2)
            loss_G2 = losses.generator_loss(fake_output_D1)

            reconstructed_images_G1 = G1(fake_images_G2, training=True)
            reconstructed_images_G2 = G2(fake_images_G1, training=True)

            loss_cycle_G1 = losses.cycle_consistency_loss(images, reconstructed_images_G1)
            loss_cycle_G2 = losses.cycle_consistency_loss(noisy_images, reconstructed_images_G2)

            total_loss_G = loss_G1 + loss_G2 + loss_cycle_G1 + loss_cycle_G2

        gradients_D1 = tape.gradient(loss_D1, D1.trainable_variables)
        gradients_D2 = tape.gradient(loss_D2, D2.trainable_variables)
        gradients_G1 = tape.gradient(total_loss_G, G1.trainable_variables)
        gradients_G2 = tape.gradient(total_loss_G, G2.trainable_variables)

        optimizer_D1.apply_gradients(zip(gradients_D1, D1.trainable_variables))
        optimizer_D2.apply_gradients(zip(gradients_D2, D2.trainable_variables))
        optimizer_G.apply_gradients(zip(gradients_G1 + gradients_G2, G1.trainable_variables + G2.trainable_variables))

        print(stats((epoch+1), (i+1), loss_D1, loss_D2, total_loss_G))
        training_logger.info(stats((epoch+1), (i+1), loss_D1, loss_D2, total_loss_G))
        if (i + 1) % 1000 == 0:
            plot_images(images, noisy_images, reconstructed_images_G1, min(5, images.shape[0]), config['general']['show_plots'], config['general']['save_plots'], (epoch+1), (i+1), config['general']['seed'])

# Main training loop
def train():
    enable_memory_growth()

    try:
        with open(CONFIGURATION_FILE, 'w') as json_file:
            json.dump(config, json_file, indent=4)
        print(f"Configurazione salvata in {CONFIGURATION_FILE}")
    except Exception as e:
        print(f"Errore nel salvataggio dell'oggetto in {CONFIGURATION_FILE}: {e}")

    train_loader, G1, G2, D1, D2, optimizer_G, optimizer_D1, optimizer_D2, noise_adder, losses = initialize_components(config)

    try:
        with open(MODEL_FILE, 'w') as file:
            file.write(f"\t\tGENERATOR 1\n{G1}\n\n")
            file.write(f"\t\tGENERATOR 2\n{G2}\n\n")
            file.write(f"\t\tDISCRIMINATOR 1\n{D1}\n\n")
            file.write(f"\t\tDISCRIMINATOR 2\n{D2}\n\n")
        print(f"Configurazione modelli salvata in {MODEL_FILE}")
    except Exception as e:
        print(f"Errore nel salvataggio in {MODEL_FILE}: {e}")

    training_logger.info("TRAINING:")
    print("\n\t\tTRAINING:")
    for epoch in range(config['general']['num_epochs']):
        train_one_epoch(config, epoch, train_loader, G1, G2, D1, D2, optimizer_G, optimizer_D1, optimizer_D2, noise_adder, losses)

if __name__ == "__main__":
    train()