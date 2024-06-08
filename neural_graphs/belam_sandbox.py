from collections import OrderedDict
from experiments.data_generalization import CNNDataset
from experiments.data import INRDataset
import json
from omegaconf import OmegaConf
from nn.relational_transformer import RelationalTransformer
import yaml
from pathlib import Path
from experiments.inr_classification.dataset.compute_mnist_statistics import compute_stats
import numpy as np


import torch
import os

dataset_path = 'experiments/bedlam/bedlam_inrs_small'
# raw_ckpt_path = 'experiments/bedlam/small 128 n_iter = 1 part_size = 1'
splits_path = "splits.json"
statistics_path = "bedlam_statistics.pth" # TODO edit
hparams_path = "./experiments/args_bedlam.yaml"
img_shape = (28, 28)
batch_size = 4

linear_layers = ['fc1', 'fc2']
conv_layers = False
prediction_name = 'init_shape'
conv_name = 'downsample_module'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_files(dataset_path, out_path):
    all_files = os.listdir(dataset_path)
    pred_v = None
    for i, file in enumerate(all_files):
        label = file.split('_')[3]
        sample_s_dict = torch.load(os.path.join(dataset_path, file))
        new_state_dict = OrderedDict()
        for k, v in sample_s_dict.items():
            layer = k.split('.')[0]
            if layer in linear_layers:
                new_state_dict[k] = v
            elif layer == prediction_name:
                pred_v = v
            if conv_layers:
                if layer == conv_name:
                    sub_layer = k.split('.')[4]
                    if sub_layer == '0':
                        new_state_dict[k] = v
        if pred_v is not None:
            new_state_dict[prediction_name] = pred_v
        name = f'{i}_{label}'
        print(name)
        torch.save(new_state_dict, os.path.join(out_path, f"{name}.ckpt"))

def create_dataset(dataset_path, split_ratio=0.9, suffix='.ckpt'):
    all_files = os.listdir(dataset_path)
    ckpt_files = [f for f in all_files if f.endswith(suffix) and not 'statistics' in f]
    np.random.shuffle(ckpt_files)
    num_train_samples = int(len(ckpt_files) * split_ratio)
    train_path = []
    val_path = []
    train_score = []
    val_score = []
    for i, file in enumerate(ckpt_files):
        label = file.rstrip(suffix).split('_')[-1]
        label = int(label == 'low')
        print(label)
        if i < num_train_samples:
            train_path.append(file)
            train_score.append(label)
        else:
            val_path.append(file)
            val_score.append(label)
    splits = {'train': {'path': train_path, 'label': train_score}, 'val': {'path': val_path, 'label': val_score}, }
    with open(f"{dataset_path}/splits.json", "w") as f:
        json.dump(splits, f)
    dataset = INRDataset(dataset_dir=f"{dataset_path}",
                             splits_path=f"splits.json",
                             )
    sample = dataset[0]
    w_shapes = [w.shape for w in sample.weights]
    return dataset

# clean_files(raw_ckpt_path, dataset_path)
train_set = create_dataset(dataset_path=dataset_path, suffix='.pth')
stats = compute_stats(data_path=dataset_path, splits_path=splits_path,
                      save_path='', statistics_path=statistics_path)
point = train_set[0]
weight_shapes = tuple(w.shape[:2] for w in point.weights)
bias_shapes = tuple(b.shape[:1] for b in point.biases)

layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
)

d_node = 64
d_edge = 32

statistics_path = (
                (Path(dataset_path) / Path(statistics_path)).expanduser().resolve()
            )
stats = torch.load(statistics_path)
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
        "sparsify": False,
        "stats": stats

    }
)

hparams = yaml.safe_load(open(hparams_path, 'r'))
optim_params = {
'lr': 1e-3,
'amsgrad':True,
'weight_decay':5e-4,
'fused':False
}

model = RelationalTransformer(layer_layout=layer_layout,
                                 graph_constructor=graph_constructor,
                                 **hparams).to(device)

batch = next(iter(train_loader)).to(device)
inputs = (batch.weights, batch.biases)
y = model(inputs)
print(y.shape)

