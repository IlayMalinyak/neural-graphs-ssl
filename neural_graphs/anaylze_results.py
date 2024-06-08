import os.path

import matplotlib.pyplot as plt
import matplotlib as mpl

from experiments.utils import *
from experiments.data import BatchSiren, INRDataset
from experiments.train import Trainer
from pathlib import Path
from omegaconf import OmegaConf
import yaml
from nn.relational_transformer import RelationalTransformer
import torch
from collections import OrderedDict

dataset_dir = r"C:\Users\Ilay\projects\geometric_dl\neural_graphs\experiments\inr_classification\dataset"
checkpoint_dir = "checkpoints/inr_classification"
splits_path = "mnist_splits.json"
statistics_path = "mnist_statistics.pth"
hparams_path = r"C:\Users\Ilay\projects\geometric_dl\neural_graphs\experiments\args.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({'font.size': 18})
def process_results():
    process_all_inr_experiments('checkpoints/inr_classification')
    process_all_siamse_experiments('checkpoints/siamse')
    proccess_all_bedlam_experiments('checkpoints/bedlam', num_gpu=1)
    fit_res = process_results_multi_gpu('checkpoints/simclr', 0, num_gpu=4, acc_as_std=True)
def predict(max_iter, exp_num, dataset_dir, checkpoint_dir, splits_path, statistics_path, hparams_path):
    if not os.path.exists(f"plots/exp{exp_num}"):
        os.makedirs(f"plots/exp{exp_num}")
    test_set = INRDataset(
            dataset_dir=dataset_dir,
            split="test",
            normalize=False,
            splits_path=splits_path,
            statistics_path=statistics_path,
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
    )
    d_node = 64
    d_edge = 32
    point = test_set[0]
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
            "sparsify": False,
            'sym_edges': False,
        }
    )

    hparams = yaml.safe_load(open(hparams_path, 'r'))
    optim_params = {
        'lr': 1e-4,
        'amsgrad': True,
        'weight_decay': 5e-4,
        'fused': False
    }

    model = RelationalTransformer(layer_layout=layer_layout,
                                  graph_constructor=graph_constructor,
                                  **hparams).to(device)
    state_dict = torch.load(f'{checkpoint_dir}/exp{exp_num}/checkpoint.ckpt')
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    state_dict = new_state_dict
    model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=None,
                      criterion=criterion, num_classes=1, hparams=hparams, optim_params=optim_params,
                      scheduler=None, train_dataloader=None,
                      val_dataloader=None, device=device, val_iter=max_iter,
                      log_path=checkpoint_dir, exp_num=exp_num)

    predicted, gt, acc, g_features, n_features, e_features, attn_weights = trainer.predict(test_loader, device)
    sum_attn_weights = attn_weights.squeeze().sum(axis=1)
    failure_indices = np.where(np.array(predicted) - np.array(gt))[0]
    sucess_indices = np.where((np.array(predicted) - np.array(gt)) == 0)[0]
    print(acc, g_features.shape, e_features.shape, attn_weights.shape, sum_attn_weights.shape)
    length_failure, strong_failure, counts_failure = [],[],[]
    length_success, strong_success, counts_success = [],[],[]
    true_labels_failures, true_labels_success = [], []
    mean_failure = None
    mean_success = None
    for i in range(max_iter // 10):
        print(i)
        idx_f = failure_indices[np.random.choice(len(failure_indices), size=1)[0]]
        name = f'{i}_fail'
        length_failure, strong_failure, counts_failure, mean_failure = process_prediction(
            attn_weights[idx_f], predicted[idx_f], gt[idx_f], layer_layout,
            length_failure, strong_failure, counts_failure, mean_failure, name, exp_num
        )
        true_labels_failures.append(gt[idx_f])


        idx_s = sucess_indices[np.random.choice(len(sucess_indices), size=1)[0]]
        name = f'{i}_success'
        length_success, strong_success, counts_success, mean_success = process_prediction(
            attn_weights[idx_s], predicted[idx_s], gt[idx_s], layer_layout,
            length_success, strong_success, counts_success, mean_success, name, exp_num
        )
        true_labels_success.append(gt[idx_s])

    df = pd.DataFrame({'true_labels': true_labels_failures,
                       'false_labels': true_labels_success,
                       'length_success': length_success,
                       'length_failure': length_failure,
                       'strong_success': strong_success,
                       'strong_failure': strong_failure,
                       'counts_success': counts_success,
                       'counts_failure': counts_failure,
                      }
                      )
    df.to_csv('plots/anaylze_results.csv')
    df_mean = pd.DataFrame({'mean_success': list(mean_success/max_iter),
                       'mean_failure': list(mean_failure/max_iter)})
    df_mean.to_csv('plots/anaylze_results_mean.csv')
    plt.scatter(true_labels_success, length_success, label='success')
    plt.scatter(true_labels_failures, length_failure, label='failure')
    plt.xlabel("label")
    plt.ylabel("mean attention length")
    plt.legend()
    plt.savefig(f'plots/exp{exp_num}/mean_length.png')
    plt.show()
    plt.scatter(true_labels_success, strong_success, label='success')
    plt.scatter(true_labels_failures, strong_failure, label='failure')
    plt.xlabel("sample")
    plt.ylabel("max_node")
    plt.legend()
    plt.savefig(f'plots/exp{exp_num}/strong.png')
    plt.show()
    plt.scatter(true_labels_success, counts_success, label='success')
    plt.scatter(true_labels_failures, counts_failure, label='failure')
    plt.xlabel("sample")
    plt.ylabel("max_node number of attentions")
    plt.legend()
    plt.savefig(f'plots/exp{exp_num}/counts.png')
    plt.show()
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(mean_success)), mean_success/max_iter, label='success', linestyle='dashed')
    plt.plot(np.arange(len(mean_failure)), mean_failure/max_iter, label='failure', linestyle='dashed')
    plt.xlabel("node")
    plt.ylabel("Mean Attention Weight")
    plt.legend()
    plt.savefig(f'plots/exp{exp_num}/mean_attn.png')
    plt.show()


def process_prediction(attn,pred, true, layer_layout,
                       length_arr, strong_arr, counts_arr, mean_arr, name, exp_num):
    attn_length, strong, counts, mean_att = plot_attn_weights(attn, pred, true,
                                                              layer_layout=layer_layout,
                                                              name=name, save_dir=f'plots/exp{exp_num}')
    counts_arr.append(counts)
    length_arr.append(attn_length)
    strong_arr.append(strong)
    if mean_arr is None:
        mean_arr = np.array(mean_att)
    else:
        mean_arr += np.array(mean_att)
    return length_arr, strong_arr, counts_arr, mean_arr


if __name__ == '__main__':
    # predict(max_iter=10000, exp_num=4, dataset_dir=dataset_dir,
    #         checkpoint_dir=checkpoint_dir,
    #         splits_path=splits_path,
    #         statistics_path=statistics_path,
    #         hparams_path=hparams_path)
    process_results()