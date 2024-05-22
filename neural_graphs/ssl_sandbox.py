import logging
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric.utils
import torchvision
from torchvision.utils import save_image, make_grid
from tqdm import trange
import wandb
import yaml

from experiments.data import INRDataset
from experiments.utils import count_parameters, set_logger, set_seed, Container
from experiments.lr_scheduler import WarmupLRScheduler
from experiments.data import BatchSiren, INRDataset, INRDatasetSSL
from experiments.transforms import *
from experiments.train import ContrastiveTrainer
from nn.relational_transformer import RelationalTransformer
from nn.ssl import SimSiam, SimCLR

from nn.gnn import GNNForClassification
from nn.dws.models import MLPModelForClassification
from nn.inr import INRPerLayer
@torch.no_grad()
def evaluate(model, loader, device, num_batches=None):
    model.eval()
    loss = 0.0
    total = 0.0
    for i, (batch1, batch2) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        inputs1 = (batch1.weights, batch1.biases)
        inputs2 = (batch2.weights, batch2.biases)
        out = model(inputs1, inputs2)
        loss += out['loss']
        total += len(batch1.label)

    model.train()
    avg_loss = loss / total
    return dict(avg_loss=avg_loss)

dataset_dir = "./experiments/inr_classification/dataset"
splits_path = "mnist_splits.json"
statistics_path = "mnist_statistics.pth"
hparams_path = "./experiments/args.yaml"
img_shape = (28, 28)
batch_size = 32

orig_mnist = torchvision.datasets.MNIST(
    Path(dataset_dir) / "mnist",
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor(),
)


dataset = INRDataset(
    dataset_dir=dataset_dir,
    split="train",
    normalize=False,
    augmentation=False,
    splits_path=splits_path,
    statistics_path=statistics_path,
)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
batch = next(iter(loader))

dataset_aug = INRDataset(
    dataset_dir=dataset_dir,
    split="train",
    normalize=False,
    augmentation=True,
    splits_path=splits_path,
    statistics_path=statistics_path,
)
loader_aug = torch.utils.data.DataLoader(
    dataset_aug, batch_size=batch_size, shuffle=False
)
# transforms = Compose([RandomTranslation(min_scale=0.1, max_scale=0.4),
#                      RandomRotate(min_deg=10, max_deg=45),
#                      RandomNoise(noise_min=1e-2, noise_max=1e-1),
#                      Dropout(p=0.8),
#                      RandomScale(min_scale=0.1, max_scale=0.4)])
transforms1 = Compose([
                        Binarize(),
                       # RandomTranslation(min_scale=0.1, max_scale=0.4),
                       RandomRotate(min_deg=10, max_deg=90),
                       RandomNoise(noise_min=1e-2, noise_max=1e-1),
                       # Dropout(p=0.01),
                       # RandomScale(min_scale=0.1, max_scale=0.4)])
                        ]
)
transforms2 = Compose([
                        Binarize(),
                       # RandomTranslation(min_scale=0.1, max_scale=0.4),
                       RandomRotate(min_deg=10, max_deg=90),
                       RandomNoise(noise_min=1e-2, noise_max=1e-1),
                       # Dropout(p=0.01),
                       # RandomScale(min_scale=0.1, max_scale=0.4)]
                        ]
)

dataset_ssl = INRDatasetSSL(dataset_dir=dataset_dir,
    split="train",
    normalize=False,
    permutation=True,
    splits_path=splits_path,
    statistics_path=statistics_path,
    transforms=[transforms1, transforms2])

loader_ssl = torch.utils.data.DataLoader(
    dataset_ssl, batch_size=batch_size, shuffle=False
)

batch_aug = next(iter(loader_aug))


batch_ssl = next(iter(loader_ssl))
# print("batch size ", torch.stack(batch_ssl).shape)
batch1, batch2 = batch_ssl.batch1, batch_ssl.batch2
batch_flatten = batch_ssl.stack()
weights = batch_flatten.weights
biases = batch_flatten.biases
labels = batch_flatten.label
print("weights: ", weights[0].shape)
shapes1 = [(w.shape, b.shape) for w, b in zip(batch1.weights, batch1.biases)]
shapes2 = [(w.shape, b.shape) for w, b in zip(batch2.weights, batch2.biases)]


print((batch1.weights[0] - batch2.weights[0]).sum())

inr_model = BatchSiren(2, 1, img_shape=img_shape)
out = inr_model(batch.weights, batch.biases)
out = out.transpose(1, 2).unflatten(2, img_shape)

out_aug = inr_model(batch_aug.weights, batch_aug.biases)
out_aug = out_aug.transpose(1, 2).unflatten(2, img_shape)

out_ssl = inr_model(weights, biases)
out_ssl = out_ssl.transpose(1, 2).unflatten(2, img_shape)
indices = [
    int(p.parts[-3].split("_")[-1]) for p in dataset.dataset["path"][:batch_size]
]
orig_images = [orig_mnist[idx][0] for idx in indices]

fig, ax = plt.subplots(1, 4)
print(labels[:batch_size])
print(labels[batch_size:])
ax[0].imshow(make_grid(orig_images).permute(1, 2, 0).clip(0, 1))
ax[0].set_title("Original Images")
ax[0].set_axis_off()

ax[1].imshow(make_grid(out).permute(1, 2, 0).clip(0, 1))
ax[1].set_title("INR reconstructions")
ax[1].set_axis_off()

ax[2].imshow(make_grid(out_ssl[:batch_size]).permute(1, 2, 0).clip(0, 1))
ax[2].set_title("Aug (1)")
ax[2].set_axis_off()

ax[3].imshow(make_grid(out_ssl[batch_size:]).permute(1, 2, 0).clip(0, 1))
ax[3].set_title("Aug (2)")
ax[3].set_axis_off()
plt.tight_layout()
plt.show()

train_set = INRDatasetSSL(
    dataset_dir=dataset_dir,
    split="train",
    normalize=False,
    splits_path=splits_path,
    statistics_path=statistics_path,
    transforms=[transforms1, transforms2],
)
val_set = INRDatasetSSL(
    dataset_dir=dataset_dir,
    split="val",
    normalize=False,
    splits_path=splits_path,
    statistics_path=statistics_path,
    transforms=[transforms1, transforms2],
)

batch_size = 32
num_workers = 2

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=batch_size,
    shuffle=False,
)

logging.info(
    f"train size {len(train_set)}, "
    f"val size {len(val_set)}, "
)
set_logger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device = {device}")

point = train_set[0][0]
weight_shapes = tuple(w.shape[:2] for w in point.weights)
bias_shapes = tuple(b.shape[:1] for b in point.biases)

layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]

logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

inr_model = None
d_node = 64
d_edge = 32

statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
stats = torch.load(statistics_path, map_location="cpu")
weights_mean = [w.mean().item() for w in stats['weights']['mean']]
weights_std = [w.mean().item() for w in stats['weights']['std']]
biases_mean = [b.mean().item() for b in stats['biases']['mean']]
biases_std = [b.mean().item() for b in stats['biases']['std']]
stats = {'weights_mean': weights_mean, 'weights_std': weights_std,
         'biases_mean': biases_mean, 'biases_std': biases_std}
print(stats)
graph_constructor = OmegaConf.create(
    {
        "_target_": "nn.graph_constructor.GraphConstructor",
        "_recursive_": False,
        "_convert_": "all",
        "d_in": 1,
        "d_edge_in": 1,
        "zero_out_bias": False,
        "zero_out_weights": False,
        "sin_emb": True,
        "sin_emb_dim": 128,
        "use_pos_embed": True,
        "input_layers": 1,
        "inp_factor": 1,
        "num_probe_features": 0,
        "inr_model": inr_model,
        "stats": stats,
    }
)

hparams = yaml.safe_load(open(hparams_path, 'r'))
optim_params = {
'lr': 1e-3,
'amsgrad':True,
'weight_decay':5e-4,
'fused':False
}

backbone = RelationalTransformer(layer_layout=layer_layout,
                                 graph_constructor=graph_constructor,
                                 **hparams)
backbone.proj_out = nn.Identity()
model_sims = SimSiam(backbone=backbone).to(device)
model_simclr = SimCLR(backbone=backbone).to(device)

parameters = [p for p in model_simclr.parameters() if p.requires_grad]
num_params = sum(p.numel() for p in model_simclr.parameters() if p.requires_grad)
print("num_params", num_params)
optimizer = torch.optim.AdamW(
    params=parameters, **optim_params)
scheduler = WarmupLRScheduler(warmup_steps=1000, optimizer=optimizer)


trainer = ContrastiveTrainer(model=model_simclr, optimizer=optimizer,
                        criterion=None, num_classes=1, hparams=hparams, optim_params=optim_params,
                       scheduler=scheduler, train_dataloader=train_loader,
                       val_dataloader=val_loader, device=device, train_iter=5)
fit_results = trainer.fit(num_epochs=1, device=device)
print(fit_results)


# epochs = 50  # doing just 50 epochs here
# eval_every = 500
#
# global_step = 0
# criterion = nn.CrossEntropyLoss()
#
# val_acc = -1
# val_loss = float("inf")
# best_val_acc = -1
# best_val_loss = float("inf")
#
# epoch_iter = trange(epochs)
# for epoch in epoch_iter:
#     for i, (batch1,batch2) in enumerate(train_loader):
#         model.train()
#         optimizer.zero_grad()
#
#         batch1 = batch1.to(device)
#         batch2 = batch2.to(device)
#         inputs1 = (batch1.weights, batch1.biases)
#         inputs2 = (batch2.weights, batch2.biases)
#         label = batch1.label
#
#         out = model(inputs1, inputs2)
#         loss = out['loss']
#         loss.backward()
#
#         optimizer.step()
#
#         if scheduler is not None:
#             scheduler.step()
#
#         epoch_iter.set_description(
#             f"[{epoch} {i+1}], train loss: {loss.item():.3f}, val loss: {val_loss:.3f},"
#             f"best val loss: {best_val_loss:.3f}, "
#         )
#
#         global_step += 1
#
#         if (global_step + 1) % eval_every == 0:
#             val_loss_dict = evaluate(model, val_loader, device)
#             val_loss = val_loss_dict["avg_loss"]