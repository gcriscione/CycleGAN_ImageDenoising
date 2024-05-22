import torch
import torch.optim as optim
from models.generators import Generator
from models.discriminators import Discriminator
from models.losses import generator_loss, discriminator_loss, cycle_consistency_loss
from utils.dataset.py import get_mnist_dataloaders
from utils.noise import add_salt_pepper_noise

def train(num_epochs=100, batch_size=1, data_path='./data'):
    train_loader = get_mnist_dataloaders(batch_size, data_path)

    G1 = Generator()
    G2 = Generator()
    D1 = Discriminator()
    D2 = Discriminator()

    optimizer_G = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D1 = optim.Adam(D1.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D2 = optim.Adam(D2.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            noisy_images = add_salt_pepper_noise(images)
            
            # Training Discriminators D1 and D2
            real_output_D1 = D1(images)
            fake_images_G2 = G2(noisy_images)
            fake_output_D1 = D1(fake_images_G2.detach())

            real_output_D2 = D2(noisy_images)
            fake_images_G1 = G1(images)
            fake_output_D2 = D2(fake_images_G1.detach())

            loss_D1 = discriminator_loss(real_output_D1, fake_output_D1)
            loss_D2 = discriminator_loss(real_output_D2, fake_output_D2)

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

            loss_G1 = generator_loss(fake_output_D2)
            loss_G2 = generator_loss(fake_output_D1)

            reconstructed_images_G1 = G1(fake_images_G2)
            reconstructed_images_G2 = G2(fake_images_G1)

            loss_cycle_G1 = cycle_consistency_loss(images, reconstructed_images_G1)
            loss_cycle_G2 = cycle_consistency_loss(noisy_images, reconstructed_images_G2)

            total_loss_G = loss_G1 + loss_G2 + loss_cycle_G1 + loss_cycle_G2

            optimizer_G.zero_grad()
            total_loss_G.backward()
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"D1 Loss: {loss_D1.item()}, D2 Loss: {loss_D2.item()}, "
                      f"G Loss: {total_loss_G.item()}")

if __name__ == "__main__":
    train()
