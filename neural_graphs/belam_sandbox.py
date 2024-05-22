from collections import OrderedDict
from experiments.data_generalization import CNNDataset
from experiments.data import INRDataset
import json

import torch
import os

dataset_path = 'experiments/bedlam/bedlam_dataset_small'
raw_ckpt_path = 'experiments/bedlam/raw'

linear_layers = ['fc1', 'fc2']
conv_layers = False
prediction_name = 'init_shape'
conv_name = 'downsample_module'

def clean_files(dataset_path, out_path):
    all_files = os.listdir(dataset_path)
    pred_v = None
    for i, file in enumerate(all_files):
        label = file.split('_')[3]
        sample_s_dict = torch.load(os.path.join(dataset_path, file))['state_dict']
        new_state_dict = OrderedDict()
        for k, v in sample_s_dict.items():
            if k.startswith('model'):
                layer = k.split('.')[2]
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

def create_dataset(dataset_path):
    all_files = os.listdir(dataset_path)
    ckpt_files = [f for f in all_files if f.endswith('ckpt')]
    train_path = []
    val_path = []
    train_score = []
    val_score = []
    for i, file in enumerate(ckpt_files):
        label = file.rstrip('.ckpt').split('_')[1]
        label = int(label == 'low')
        print(label)
        if i < 8:
            train_path.append(file)
            train_score.append(1)
        else:
            val_path.append(file)
            val_score.append(1)
    splits = {'train': {'path': train_path, 'label': train_score}, 'val': {'path': val_path, 'score': val_score}, }
    with open(f"{dataset_path}/splits.json", "w") as f:
        json.dump(splits, f)
    cnn_dataset = INRDataset(dataset_dir=f"{dataset_path}",
                             splits_path=f"splits.json",
                             )
    sample = cnn_dataset[0]
    w_shapes = [w.shape for w in sample.weights]
    print(w_shapes)

# clean_files(raw_ckpt_path, dataset_path)
create_dataset(dataset_path=dataset_path)