from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Class to create a DataLoader for the MNIST dataset
class MNISTDataLoader:
    def __init__(self, batch_size=1, data_path='./data'):
        self.batch_size = batch_size
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),                  # Convert the image to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))    # Normalize the images with mean 0.5 and std 0.5
        ])

    # Creates and returns a DataLoader for the MNIST training dataset
    def get_dataloader(self):
        train_dataset = datasets.MNIST(root=self.data_path, train=True, transform=self.transform, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader