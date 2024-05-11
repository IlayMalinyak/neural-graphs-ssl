import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # print("projection_MLP: ", x.shape)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=64):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def info_nce_loss(features, temperature):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    return nll

class SimCLR(nn.Module):
    def __init__(self, backbone, hidden_dim=64):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.num_graph_features,
                                        hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.predictor = prediction_MLP(in_dim=hidden_dim,
                                        hidden_dim=hidden_dim//2, out_dim=hidden_dim)
    def forward(self, x, temperature=1.0):
        z = self.backbone(x)
        p = self.projector(z)
        L = info_nce_loss(z, temperature=temperature)
        return {'loss': L, 'features': z, 'predictions': p}


def SiamseLoss(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projector = projection_MLP(backbone.num_graph_features)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, inputs):
        b_size = inputs[0][0].shape[0]
        f, h = self.encoder, self.predictor
        z = f(inputs)
        z1, z2 = z[:b_size//2], z[b_size//2:]
        p1, p2 = h(z1), h(z2)
        L = SiamseLoss(p1, z2) / 2 + SiamseLoss(p2, z1) / 2
        return {'loss': L, 'features': (z1, z2), 'predictions': (p1, p2)}