import os
from data.download_mist import download_mnist
from train import train

if __name__ == "__main__":
    data_path = './data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    download_mnist(data_path)
    train(data_path=data_path)