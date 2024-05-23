import torch.optim as optim
from models.generators import Generator
from models.discriminators import Discriminator
from models.losses import Losses
from utils.dataset import MNISTDataLoader
from utils.noise import NoiseAdder
from utils.plot import plot_images
from utils.stats import print_stats
from config.config_file import config




def print_config(obj):
    for key, value in obj.items():
        print(f"{key}: {value}")

def train():    
    data_loader = MNISTDataLoader(batch_size=config['general']['batch_size'], data_path=config['general']['data_path'])
    train_loader = data_loader.get_dataloader()

    print("\t\tCONFIGURATION:")
    print("\nGeneral")
    print_config(config['general'])
    print("\nNoise_adder")
    print_config(config['noise_adder'])
    print("\nTraining")
    print_config(config['training'])
    print("")
    G1 = Generator(config['generator'])
    G2 = Generator(config['generator'])
    D1 = Discriminator(config['discriminator'])
    D2 = Discriminator(config['discriminator'])

    print(f"\t\tGENERATOR 1\n{G1}\n")
    print(f"\t\tGENERATOR 2\n{G1}\n")
    print(f"\t\tDISCRIMINATOR 1\n{D1}\n")
    print(f"\t\tDISCRIMINATOR 2\n{D1}\n")

    optimizer_G = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=config['general']['learning_rate'], betas=(config['general']['beta1'], config['general']['beta2']))
    optimizer_D1 = optim.Adam(D1.parameters(), lr=config['general']['learning_rate'], betas=(config['general']['beta1'], config['general']['beta2']))
    optimizer_D2 = optim.Adam(D2.parameters(), lr=config['general']['learning_rate'], betas=(config['general']['beta1'], config['general']['beta2']))

    noise_adder = NoiseAdder(noise_type=config['noise_adder']['noise_type'], noise_ratio=config['noise_adder']['salt_pepper_ratio'])
    losses = Losses(loss_type=config['training']['loss_function'])

    for epoch in range(config['general']['num_epochs']):
        for i, (images, _) in enumerate(train_loader):
            noisy_images = noise_adder.add_noise(images)
            
            # Training Discriminators D1 and D2
            real_output_D1 = D1(images)
            fake_images_G2 = G2(noisy_images)
            fake_output_D1 = D1(fake_images_G2.detach())

            real_output_D2 = D2(noisy_images)
            fake_images_G1 = G1(images)
            fake_output_D2 = D2(fake_images_G1.detach())

            loss_D1 = losses.discriminator_loss(real_output_D1, fake_output_D1)
            loss_D2 = losses.discriminator_loss(real_output_D2, fake_output_D2)

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            loss_D1.backward()
            loss_D2.backward()
            optimizer_D1.step()
            optimizer_D2.step()

            # Training Generators G1 and G2
            fake_images_G1 = G1(images)
            fake_images_G2 = G2(noisy_images)

            fake_output_D1 = D1(fake_images_G2)
            fake_output_D2 = D2(fake_images_G1)

            loss_G1 = losses.generator_loss(fake_output_D2)
            loss_G2 = losses.generator_loss(fake_output_D1)

            reconstructed_images_G1 = G1(fake_images_G2)
            reconstructed_images_G2 = G2(fake_images_G1)

            loss_cycle_G1 = losses.cycle_consistency_loss(images, reconstructed_images_G1)
            loss_cycle_G2 = losses.cycle_consistency_loss(noisy_images, reconstructed_images_G2)

            total_loss_G = loss_G1 + loss_G2 + loss_cycle_G1 + loss_cycle_G2

            optimizer_G.zero_grad()
            total_loss_G.backward()
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print_stats(epoch, i + 1, loss_D1, loss_D2, total_loss_G)
                plot_images(images, noisy_images, reconstructed_images_G1, n_images=5)

if __name__ == "__main__":
    train()
