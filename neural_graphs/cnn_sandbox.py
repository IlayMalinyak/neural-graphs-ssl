
from experiments.cnn_generalization.dataset.cnn_sampler import CNNConfig, CNN, DEFAULT_CONFIG_OPTIONS, ACTIVATION_FN
from experiments.data_generalization import CNNDataset
import torch
import json
import numpy as np

dataset_dir = 'experiments/cnn_generalization/dataset/toy_example_dataset'

cfg = CNNConfig(n_layers=4,
    n_classes=10,
    channels=[3,16,32,64,128],
    kernel_size=[3, 5,7,7,],
    stride=[1,1,1,1],
    padding=[1,1,1,1],
    residual=[-1,-1,0,0],
    activation=['none', 'relu', 'relu', 'relu'],)
train_path = []
val_path = []
train_score = []
val_score = []
for i in range(5):
    cnn = CNN(cfg)
    print(cnn)
    state_dict = cnn.state_dict()
    torch.save(cnn.state_dict(), f'{dataset_dir}/checkpoints/cnn_{i}.ckpt')
    if i < 4:
        train_path.append(f'checkpoints/cnn_{i}.ckpt')
        train_score.append(np.random.randint(0, 100))
    else:
        val_path.append(f'checkpoints/cnn_{i}.ckpt')
        val_score.append(np.random.randint(0, 100))


splits = {'train': {'path': train_path, 'score': train_score}, 'val': {'path': val_path, 'score': val_score}, }
with open(f"{dataset_dir}/splits.json", "w") as f:
    json.dump(splits, f)


cnn_dataset = CNNDataset(dataset_dir=f"{dataset_dir}",
                         splits_path=f"splits.json",
                        max_kernel_size=(9, 9)
                         )

print(cnn_dataset[0])
