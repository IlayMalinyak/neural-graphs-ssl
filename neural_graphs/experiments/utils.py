import argparse
import logging
import os
import random
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import json
from torch.distributed import init_process_group

common_parser = argparse.ArgumentParser(add_help=False, description="common parser")
common_parser.add_argument("--data-path", type=str, help="path for dataset")
common_parser.add_argument("--save-path", type=str, help="path for output file")


class Container(object):
    '''A container class that can be used to store any attributes.'''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def load_dict(self, dict):
        for key, value in dict.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def print_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def plot_fit(
    fit_res: dict,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
    acc_label: str = 'Accuracy (%)',
    acc_ticks: tuple = None,
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 1 if np.isnan(fit_res['train_acc']).any() else 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()
    if ncols > 1:
        p = itertools.product(enumerate(["train", "val"]), enumerate(["loss", "acc"]))
    else:
        p = itertools.product(enumerate(["train", "val"]), enumerate(["loss"]))
    for (i, traintest), (j, lossacc) in p:
        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data =fit_res[attr]
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        # ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel(acc_label)
            if acc_ticks is not None:
                ax.set_yticks(acc_ticks[0])
                ax.set_yticklabels(acc_ticks[1])

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes

def process_results_multi_gpu(checkpoint_dir, exp_num, num_gpu=4, acc_as_std=False):
    output_filename = f'{checkpoint_dir}/exp{exp_num}/fit_results.json'
    with open(output_filename, "r") as f:
        fit_results = json.load(f)
    fit_results['train_acc'] = np.array(fit_results['train_acc']) / num_gpu
    fit_results['val_acc'] = np.array(fit_results['val_acc']) / num_gpu
    if acc_as_std:
        fig, axes = plot_fit(fit_results, legend=exp_num, train_test_overlay=True,
                             acc_label='std', acc_ticks=([0.08, 0.125, 0.15], ['', r'$\frac{1}{\sqrt{d}}$', '']))
    else:
        fig, axes = plot_fit(fit_results, legend=exp_num, train_test_overlay=True,)
    plt.savefig(f"{checkpoint_dir}/exp{exp_num}/fit.png")
