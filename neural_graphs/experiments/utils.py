import argparse
import logging
import os
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import json
from torch.distributed import init_process_group
import seaborn as sn
import matplotlib.patches as patches


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


def plot_features(g_features, e_features, n_features, prediction, gt, name, save_dir):
    fig, ax = plt.subplots(1,2)
    ax[0].plot(g_features.squeeze(), label="g_features")
    ax[0].plot(n_features.squeeze().sum(axis=-1), label="n_features summed")
    # ax[0].legend()
    ax[1].imshow(e_features.squeeze().sum(axis=-1), label="e_features summed")
    # ax[1].legend()
    ax[0].grid()
    fig.suptitle(f"Prediction: {prediction}, GT: {gt}")
    plt.savefig(os.path.join(save_dir, f"{name}_features.png"))
    plt.close()
    return fig, ax

def plot_attn_weights(attn, prediction, gt, layer_layout, num_heads=8, name='attn', save_dir='plots'):
    summed_attn = attn.squeeze().sum(axis=0)
    summed_attn = (summed_attn - summed_attn.min()) / (summed_attn.max() - summed_attn.min())
    ax = sn.heatmap(summed_attn)
    # Add a white rectangle patch for the area rows 0:2 and columns 0:2
    rect1 = patches.Rectangle((0, 0), 2, 2, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(rect1)

    # Add a white rectangle patch for the area rows 2:34 and columns 2:34
    rect2 = patches.Rectangle((2, 2), 32, 32, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(rect2)
    rect2 = patches.Rectangle((34, 34), 32, 32, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(rect2)
    plt.savefig(os.path.join(save_dir, f"{name}_attn_weights.png"))
    plt.close()

    max_attn = np.argmax(summed_attn, axis=-1)
    attn_length = max_attn - np.arange(len(max_attn))
    node_in = 0
    node_out = layer_layout[0]
    non_local_attention = np.zeros_like(max_attn)
    fig, axes = plt.subplots(1,len(layer_layout)-1, figsize=(26,8))
    for i, l in enumerate(layer_layout[1:]):
        layer_attn = summed_attn[node_in:node_out, node_in:node_out]
        sn.heatmap(layer_attn, ax=axes[i])
        axes[i].set_title(f'Layer {i}')

        forward_mask = max_attn[node_in:node_out] > node_out
        backward_mask = max_attn[node_in:node_out] < node_in
        non_local_attention[node_in:node_out] = np.logical_or(forward_mask, backward_mask)
        node_in = node_out
        node_out = node_out + l

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_attn_weights_layers.png"))
    plt.close()
    unique, counts = np.unique(max_attn, return_counts=True)
    max_counts = counts.max()
    strong_node = unique[np.argmax(counts)]
    mean_attn = summed_attn.mean(axis=0)
    return attn_length.mean(), strong_node, max_counts, mean_attn


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
                             acc_label='std', acc_ticks=([0,0.08, 0.125, 0.15],
                                                         ['0', '', r'$\frac{1}{\sqrt{d}}$', '']))
    else:
        fig, axes = plot_fit(fit_results, legend=exp_num, train_test_overlay=True,)
    plt.savefig(f"{checkpoint_dir}/exp{exp_num}/fit.png")
    plt.close()
    return fit_results

def aggregate_loss_per_epoch(loss_list, acc_list):
    num_epochs = len(acc_list)
    iterations_per_epoch = len(loss_list) // num_epochs
    validation_loss = np.array(loss_list)
    loss_matrix = validation_loss[:iterations_per_epoch*num_epochs].reshape(num_epochs, iterations_per_epoch)
    average_validation_loss_per_epoch = np.mean(loss_matrix, axis=1)
    average_validation_loss_per_epoch = average_validation_loss_per_epoch.tolist()
    return average_validation_loss_per_epoch
def process_all_inr_experiments(checkpoint_dir, num_gpu=4, save_dir='plots'):
    with open(f'{checkpoint_dir}/transforms.json', 'r') as f:
        transforms_dict = json.load(f)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    max_loss = 0
    acc_df = pd.DataFrame()
    for dir in os.listdir(checkpoint_dir):
        print(dir)
        if os.path.isdir(os.path.join(checkpoint_dir, dir)) and dir.startswith("exp"):
            transf = transforms_dict[dir]
            preds_df = pd.read_csv(os.path.join(checkpoint_dir, dir, 'test_preds.csv'))
            acc = preds_df['preds'].eq(preds_df['gt']).sum() / len(preds_df)
            acc_df[transf] = [acc]
            exp_num = dir[-1]
            fit_res = process_results_multi_gpu(checkpoint_dir, exp_num, num_gpu=num_gpu, acc_as_std=False)
            data_loss = fit_res['val_loss']
            data_acc = fit_res['val_acc']
            epoch_loss = aggregate_loss_per_epoch(data_loss, data_acc)
            max_loss = max(max_loss, max(epoch_loss))
            axes[0].plot(np.arange(1, len(epoch_loss) + 1), epoch_loss, label=transf)
            axes[0].legend(fontsize=12)
            axes[0].grid(True)
            axes[0].set_ylabel('CE Loss')
            axes[0].set_xlabel('Epoch #')
            axes[0].set_ylim([0, max_loss])
            axes[1].plot(np.arange(1, len(data_acc) + 1), data_acc)
            axes[1].grid(True)
            axes[1].set_ylabel('Accuracy')
            axes[1].set_xlabel('Epoch #')
            axes[1].set_ylim([0, 1])
            plt.tight_layout()
    plt.savefig(f'{save_dir}/inr_results.png')
    acc_df.to_csv(f'{checkpoint_dir}/acc_df.csv')
    plt.show()

def process_all_siamse_experiments(checkpoint_dir, num_gpu=4, save_dir='plots'):
    with open(f'{checkpoint_dir}/transforms.json', 'r') as f:
        transforms_dict = json.load(f)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    max_loss = 0
    for dir in os.listdir(checkpoint_dir):
        if os.path.isdir(os.path.join(checkpoint_dir, dir)):
            transf = transforms_dict[dir]
            exp_num = dir[-1]
            fit_res = process_results_multi_gpu(checkpoint_dir, exp_num, num_gpu=num_gpu, acc_as_std=True)
            data_loss = fit_res['val_loss']
            data_acc = fit_res['val_acc']
            axes[0].plot(np.arange(1, len(data_loss) + 1), data_loss, label=transf)
            axes[0].legend(fontsize=12)
            axes[0].grid(True)
            axes[0].set_ylabel('SimSiam Loss')
            axes[0].set_xlabel('Iteration #')
            axes[1].plot(np.arange(1, len(data_acc) + 1), data_acc)
            axes[1].grid(True)
            axes[1].set_ylabel('Std')
            axes[1].set_xlabel('Epoch #')
            std_ticks = ([0, 0.08, 0.125, 0.15], ['0', '', r'$\frac{1}{\sqrt{d}}$', ''])
            axes[1].set_yticks(std_ticks[0])
            axes[1].set_yticklabels(std_ticks[1])
            plt.tight_layout()
    plt.savefig(f'{save_dir}/siamse_results.png')
    plt.show()

def proccess_all_bedlam_experiments(checkpoint_dir, num_gpu=1, save_dir='plots'):
    with open(f'{checkpoint_dir}/exps.json', 'r') as f:
        exps = json.load(f)
    max_loss = 0
    for dir in os.listdir(checkpoint_dir):
        if os.path.isdir(os.path.join(checkpoint_dir, dir)):
            exp_num = dir[-1]
            label = exps[dir]
            fit_res = process_results_multi_gpu(checkpoint_dir, exp_num, num_gpu=num_gpu, acc_as_std=True)
            data_loss = fit_res['val_loss']
            data_acc = fit_res['val_acc']
            # axes[0].plot(np.arange(1, len(data_loss) + 1), data_loss)
            # axes[0].legend(fontsize=12)
            # axes[0].grid(True)
            # axes[0].set_ylabel('CE Loss')
            # axes[0].set_xlabel('Iteration #')
            # axes[1].plot(np.arange(1, len(data_acc) + 1), data_acc)
            # axes[1].grid(True)
            # axes[1].set_ylabel('Accuracy')
            # axes[1].set_xlabel('Epoch #')
            plt.plot(np.arange(1, len(data_acc) + 1), data_acc, label=label)
            plt.grid(True)
            plt.ylabel('Validation Accuracy')
            plt.xlabel('Epoch #')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/bedlam_results.png')
    plt.show()

