import torch
import os
import yaml
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F

class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """

    def __init__(self, model, optimizer, criterion, train_dataloader, device, world_size=1,
                 scheduler=None, val_dataloader=None, hparams=None, optim_params=None, train_iter=np.inf,
                 val_iter=np.inf, scaler=None, grad_clip=False, num_classes=1,
                 exp_num=None, log_path=None, exp_name=None, plot_every=None, ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.device = device
        self.num_classes = num_classes
        self.world_size = world_size
        self.optim_params = optim_params
        self.net_params = hparams
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        self.logger = None
        if not os.path.exists(f'{self.log_path}/exp{self.exp_num}'):
            os.makedirs(f'{self.log_path}/exp{self.exp_num}')
        # with open(f'{self.log_path}/exp{exp_num}/net_params.yml', 'w') as outfile:
        #     yaml.dump(self.net_params, outfile, default_flow_style=False)
        # with open(f'{self.log_path}/exp{exp_num}/optim_params.yml', 'w') as outfile:
        #     yaml.dump(self.optim_params, outfile, default_flow_style=False)

    def fit(self, num_epochs, device, early_stopping=None, best='loss'):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss, = [], []
        train_acc, val_acc = [], []
        # self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        if torch.distributed.is_initialized():
            main_process = torch.distributed.get_rank() == 0
        else:
            main_process = True

        print(f"Starting training for {num_epochs} epochs with parameters:"
              f" {self.optim_params}, {self.net_params}")
        print("is main process: ", main_process, flush=True)
        for epoch in range(num_epochs):
            start_time = time.time()
            t_loss, t_acc = self.train_epoch(device)
            t_loss_mean = np.mean(t_loss)
            train_loss.extend(t_loss)
            global_train_accuracy, global_train_loss = self.process_loss(t_acc, t_loss_mean)
            if main_process:  # Only perform this on the master GPU
                train_acc.append(global_train_accuracy.mean().item())

            v_loss, v_acc = self.eval_epoch(device)
            v_loss_mean = np.mean(v_loss)
            val_loss.extend(v_loss)
            global_val_accuracy, global_val_loss = self.process_loss(v_acc, v_loss_mean)
            if main_process:  # Only perform this on the master GPU
                val_acc.append(global_val_accuracy.mean().item())
                if self.scheduler is not None:
                    self.scheduler.step(global_val_loss)
                criterion = min_loss if best == 'loss' else best_acc
                mult = 1 if best == 'loss' else -1
                objective = global_val_loss if best == 'loss' else global_val_accuracy.mean()
                if mult * objective < mult * criterion:
                    print("saving model...")
                    if best == 'loss':
                        min_loss = global_val_loss
                    else:
                        best_acc = global_val_accuracy.mean()
                    torch.save(self.model.state_dict(), f'{self.log_path}/exp{self.exp_num}/checkpoint.ckpt')
                    self.best_state_dict = self.model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == early_stopping:
                        print('early stopping!', flush=True)
                        break
                print(f'Epoch {epoch}: Train Loss: {global_train_loss:.6f}, Val Loss:' \
                      f'{global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, ' \
                      f'Val Acc: {global_val_accuracy.round(decimals=4).tolist()}, Time: {time.time() - start_time:.2f}s',
                      flush=True)
                if epoch % 10 == 0:
                    print(os.system('nvidia-smi'))

        self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']
        # self.optim_params['lr_history'].append(self.optim_params['lr'])
        with open(f'{self.log_path}/exp{self.exp_num}/optim_params.yml', 'w') as outfile:
            yaml.dump(self.optim_params, outfile, default_flow_style=False)
        return {"num_epochs": num_epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc,
                "val_acc": val_acc}
    def process_loss(self, acc, loss_mean):
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            global_accuracy = torch.tensor(acc).cuda() if acc is not None else None # Convert accuracy to a tensor on the GPU
            torch.distributed.reduce(global_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss = torch.tensor(loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss /= torch.distributed.get_world_size()
        else:
            global_loss = torch.tensor(loss_mean)
            global_accuracy = torch.tensor(acc)
        return global_accuracy, global_loss

    def train_epoch(self, device):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        correct = 0
        total = 0
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            self.model.train()
            self.optimizer.zero_grad()

            batch = batch.to(device)
            inputs = (batch.weights, batch.biases)
            label = batch.label

            out = self.model(inputs)
            loss = self.criterion(out, label)
            loss.backward()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            pred = out.argmax(1)
            correct += pred.eq(batch.label).sum()
            total += len(batch.label)
            if i > self.train_iter:
                break
        return train_loss, correct / total

    def eval_epoch(self, device):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        correct = 0.0
        total = 0.0
        predicted, gt = [], []
        pbar = tqdm(self.val_dl)
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            inputs = (batch.weights, batch.biases)
            out = self.model(inputs)
            val_loss.append(F.cross_entropy(out, batch.label, reduction="sum"))
            total += len(batch.label)
            pred = out.argmax(1)
            correct += pred.eq(batch.label).sum()
            predicted.extend(pred.cpu().numpy().tolist())
            gt.extend(batch.label.cpu().numpy().tolist())
            if i > self.val_iter:
                break
        return val_loss, correct / total


class ContrastiveTrainer(Trainer):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    def train_epoch(self, device):
        self.model.train()
        train_loss = []
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            self.model.train()
            self.optimizer.zero_grad()

            batch = batch.to(device)
            batch = batch.stack()
            inputs = (batch.weights, batch.biases)

            out = self.model(inputs)
            loss = out['loss']
            loss.backward()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            train_loss.append(loss.item())
            if i > self.train_iter:
                break
        return train_loss, np.nan

    @torch.no_grad()
    def eval_epoch(self, device):
        self.model.eval()
        val_loss = []
        pbar = tqdm(self.val_dl)
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            batch= batch.stack()
            inputs = (batch.weights, batch.biases)
            out = self.model(inputs)
            loss = out['loss'].item()
            val_loss.append(loss)
            if i > self.val_iter:
                break
        return val_loss, np.nan