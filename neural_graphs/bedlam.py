import logging
import os
from pathlib import Path
import pandas as pd
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
from collections import OrderedDict
# import wandb
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)  

from experiments.data import INRDataset
from experiments.utils import count_parameters, set_logger, set_seed, plot_fit
from experiments.lr_scheduler import WarmupLRScheduler
from experiments.data import BatchSiren, INRDataset, INRDatasetSSL, build_dataset
from experiments.transforms import *
from experiments.train import Trainer
from nn.relational_transformer import RelationalTransformer
from nn.ssl import SimSiam

dataset_dir = "/data/neural_graphs/experiments/bedlam/bedlam_dataset"
checkpoint_dir = "/data/neural_graphs/checkpoints/bedlam"
splits_path = "splits.json"
statistics_path = "bedlam_statistics.pth"
hparams_path = "/data/neural_graphs/experiments/args.yaml"
img_shape = (28, 28)
batch_size = 1
exp_num = 1
num_epochs = 200

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("logdir", checkpoint_dir)

if __name__ == "__main__":
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    setup(rank, world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")

    train_set = INRDataset(
    dataset_dir=dataset_dir,
    split="train",
    normalize=False,
    augmentation=True,
    splits_path=splits_path,
    statistics_path=statistics_path,
    )
    val_set = INRDataset(
        dataset_dir=dataset_dir,
        split="val",
        normalize=False,
        splits_path=splits_path,
        statistics_path=statistics_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=slurm_cpus_per_task,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=slurm_cpus_per_task,
        pin_memory=True
    )

    print(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
    )
    print(f"device = {device}")

    point = train_set[0]
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
    'lr': 5e-4,
    'amsgrad':True,
    'weight_decay':5e-4,
    'fused':False
    }

    model = RelationalTransformer(layer_layout=layer_layout,
                                     graph_constructor=graph_constructor,
                                     **hparams)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    parameters = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num_params", num_params)
    optimizer = torch.optim.AdamW(
        params=parameters, **optim_params)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=optimizer,
                            criterion=criterion, num_classes=1, hparams=hparams, optim_params=optim_params,
                           scheduler=None, train_dataloader=train_loader,
                           val_dataloader=val_loader, device=device,
                            log_path=checkpoint_dir, exp_num=exp_num)
    fit_results = trainer.fit(num_epochs=num_epochs, device=device, early_stopping=20)
    output_filename = f'{checkpoint_dir}/exp{exp_num}/fit_results.json'
    with open(output_filename, "w") as f:
        json.dump(fit_results, f, indent=2)
    fig, axes = plot_fit(fit_results, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{checkpoint_dir}/exp{exp_num}/fit.png")

    preds, gt, test_acc = trainer.predict(val_loader, local_rank)
    print(f"Test accuracy: {test_acc}")
    df = pd.DataFrame({"preds": preds, "gt": gt})
    df.to_csv(f"{checkpoint_dir}/exp{exp_num}/test_preds.csv", index=False)