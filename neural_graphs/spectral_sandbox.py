import logging
from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch_geometric.utils
import torchvision
from torch_geometric.data import DataLoader

from experiments.data_generalization import SpectralDataset
from nn.spectral_gcnn import ChebConvNet

dataset_dir = "./experiments/inr_classification/dataset"
splits_path = "mnist_splits.json"
statistics_path = "mnist_statistics.pth"
hparams_path = "./experiments/args.yaml"
img_shape = (28, 28)
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

orig_mnist = torchvision.datasets.MNIST(
    Path(dataset_dir) / "mnist",
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)


dataset = SpectralDataset(
    dataset_dir=dataset_dir,
    split="train",
    normalize=False,
    splits_path=splits_path,
    statistics_path=statistics_path,
)

sample = dataset[0]

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = ChebConvNet(in_channels=1,
                 hidden_channel=32,
                 out_channels=64,
                 num_hiddens=4,
                 d_out=10,
                 d_out_hid=16,
                 dropout=0.3,).to(device)
x = next(iter(loader))
y = model(x.to(device))
print(y.shape)