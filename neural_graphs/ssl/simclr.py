import logging
import os
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json


os.chdir("../")
from experiments.data import INRDataset
from experiments.utils import count_parameters, set_logger, set_seed, plot_fit
from experiments.lr_scheduler import WarmupLRScheduler
from experiments.data import BatchSiren, INRDataset, INRDatasetSSL, build_dataset
from experiments.transforms import *
from experiments.train import ContrastiveTrainer
from nn.relational_transformer import RelationalTransformer
from nn.ssl import SimCLR

dataset_dir = r"C:\Users\Ilay\projects\geometric_dl\neural-graphs\experiments\inr_classification\dataset"
checkpoint_dir = "./checkpoints/simclr"
splits_path = "mnist_splits.json"
statistics_path = "mnist_statistics.pth"
hparams_path = "./experiments/args.yaml"
img_shape = (28, 28)
batch_size = 32
exp_num = 0

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if torch.cuda.is_available():
        print("Using GPU")
        world_size = os.environ.get("WORLD_SIZE", 1)
        rank = os.environ.get("PROCID", 0)
        gpus_per_node = torch.cuda.device_count()
        if "SLURM_JOB_ID" in os.environ and "SLURM_NODELIST" in os.environ:
            setup(rank, world_size)
            if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        print(f"rank: {rank}, local_rank: {local_rank}")
    else:
        local_rank = device
        world_size = 1
        rank = 0


    train_set, val_set = build_dataset(INRDatasetSSL, dataset_dir, splits_path, statistics_path)

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

    model = SimCLR(backbone=backbone).to(device)
    if "SLURM_JOB_ID" in os.environ and "SLURM_NODELIST" in os.environ:
        model = DDP(model, device_ids=[local_rank])

    parameters = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num_params", num_params)
    optimizer = torch.optim.AdamW(
        params=parameters, **optim_params)
    scheduler = WarmupLRScheduler(warmup_steps=1000, optimizer=optimizer)


    trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                            criterion=None, num_classes=1, hparams=hparams, optim_params=optim_params,
                           scheduler=scheduler, train_dataloader=train_loader,
                           val_dataloader=val_loader, device=device, train_iter=5, val_iter=5,
                            log_path=checkpoint_dir, exp_num=exp_num)
    fit_results = trainer.fit(num_epochs=1, device=device)
    output_filename = f'{checkpoint_dir}/exp{exp_num}/fit_results.json'
    with open(output_filename, "w") as f:
        json.dump(fit_results, f, indent=2)
    print(fit_results)
    fig, axes = plot_fit(fit_results, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{checkpoint_dir}/exp{exp_num}/fit.png")

