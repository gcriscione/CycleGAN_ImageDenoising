import torchvision.datasets as datasets

# Download MNIST dataset in specified path
def download_mnist(data_path='./data'):
    datasets.MNIST(root=data_path, train=True, download=True)
    datasets.MNIST(root=data_path, train=False, download=True)


# ----------------------------
if __name__ == "__main__":
    download_mnist()