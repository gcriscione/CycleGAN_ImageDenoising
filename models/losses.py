import torch
import torch.nn as nn



# Loss functions for the GAN
class Losses:
    def __init__(self, loss_type='MSE'):
        if loss_type == 'MSE':
            self.adversarial_loss = nn.MSELoss()
        elif loss_type == 'BCE':
            self.adversarial_loss = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.cycle_loss = nn.L1Loss()

    # Generator loss
    def generator_loss(self, fake_output):
        return self.adversarial_loss(fake_output, torch.ones_like(fake_output))

    # Discriminator loss
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.adversarial_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.adversarial_loss(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2

    # Cycle consistency loss
    def cycle_consistency_loss(self, real_image, reconstructed_image):
        return self.cycle_loss(real_image, reconstructed_image)
