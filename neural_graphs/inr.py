import os

from nn.inr import INR
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from experiments.utils import make_coordinates
from experiments.data import BatchSiren

labels_dict = {'low':0, 'high':1}
imgs_path = 'experiments/bedlam/imgs_high/png'
inr_path = 'experiments/bedlam/bedlam_inrs_small'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def list_all_files(root_folder):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, img_shape=None, dims=2):
        super().__init__()
        self.imgs_path = imgs_path
        self.imgs = []
        self.img_shape = img_shape
        self.dims=dims
        self.out_dim = 1 if self.dims==2 else self.dims

    def __len__(self):
        return len(self.imgs_path)
    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        if self.img_shape is not None:
            img = img.resize(self.img_shape)
        if self.dims > 2:
            img = img.convert('RGB')
            img = torch.from_numpy(np.array(img).transpose((2, 0, 1)))
        else:
            img = img.convert('L')
            img = torch.from_numpy(np.array(img))
        img_shape = self.img_shape if self.img_shape is not None else img.size[-2:]
        coords = make_coordinates(img_shape, bs=1).squeeze()
        values = img.reshape(self.out_dim, -1).T.float()
        return img, coords, values

def create_dataset(num_iters=1000, img_shape=(128,224), dims=3, plot_every=100, num_samples=np.inf, offset=0):
    all_paths = list_all_files(imgs_path)
    ds = ImageDataset(all_paths, img_shape=img_shape, dims=dims)
    label = imgs_path.split('/')[-2].split('_')[-1]
    print(len(ds))

    criterion = torch.nn.MSELoss()
    for i, (img, coords, values) in enumerate(ds):
        values = values.to(device)
        model = INR(hidden_features=128, n_layers=3,
                    out_features=dims).to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-3)
        pbar = tqdm(range(num_iters))
        losses = []
        for t in pbar:
            optimizer.zero_grad()
            pred_values = model(coords.to(device)).float()
            loss = criterion(pred_values, values)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss.item()}')
            losses.append(loss.item())
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(inr_path, f'sample_{i+offset}_label_{label}.pth'))
        print(f'Sample {i+offset} label {label} saved in {inr_path}')
        if i % plot_every == 0:
            plot_results(dims, i+offset, img, label, losses, model, pred_values)
        if i >= num_samples:
            break


def plot_results(dims, i, img, label, losses, model, pred_values):
    pred_values = pred_values.transpose(-1, -2).unflatten(-1, img.shape[-2:]).squeeze(0).cpu().detach().numpy()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    if dims >= 2:
        axes[0].imshow(img.numpy().transpose((1, 2, 0)))
        axes[1].imshow(pred_values.transpose((1, 2, 0)))
    else:
        axes[0].imshow(img.numpy().squeeze(), cmap='gray')
        axes[1].imshow(pred_values.squeeze(), cmap='gray')
    axes[0].set_title('Original')
    axes[1].set_title('Reconstruction')
    plt.savefig(f'plots/bedlam_recon/sample_{i}_dims_{dims}_img.png')
    plt.show()
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction MSE Error')
    plt.savefig(f'plots/bedlam_recon/sample_{i}_dims_{dims}_loss.png')
    plt.show()


def test_reconstructions():
    all_paths = [os.path.join(imgs_path, f) for f in os.listdir(imgs_path)]
    all_labels = [1] * len(all_paths)
    ds = ImageDataset(all_paths, img_shape=(128,64))
    checkpoints = [os.path.join(inr_path, f) for f in os.listdir(inr_path)]
    for checkpoint in checkpoints:
        sample = int(checkpoint.removesuffix('.pth').split('_')[-3])
        img = ds[sample][0].numpy()
        state_dict = torch.load(checkpoint)
        weights = tuple(
            [v.unsqueeze(0).unsqueeze(-1).cpu() for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v.unsqueeze(0).unsqueeze(-1).cpu() for w, v in state_dict.items() if "bias" in w])
        inr_model = BatchSiren(2, 3, img_shape=img.shape)
        out = inr_model(weights, biases)
        out = out.transpose(1, 2).unflatten(2, img.shape).squeeze(0).numpy()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        axes[0].imshow(img.squeeze(), cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(out.squeeze(), cmap='gray')
        axes[1].set_title('Reconstruction')
        plt.show()

        print(out.shape, img.shape)


if __name__ == '__main__':
    num_ready = len(os.listdir(inr_path))
    create_dataset(num_iters=15000, dims=1, img_shape=(64,64), num_samples=1000, offset=num_ready)



