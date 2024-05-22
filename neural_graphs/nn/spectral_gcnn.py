from torch_geometric.nn import ChebConv
from torch_geometric.utils import scatter

from nn.pooling import HomogeneousAggregator
import torch
import torch.nn as nn



class ChebConvNet(nn.Module):
    def __init__(self, in_channels,
                 hidden_channel,
                 out_channels,
                 num_hiddens,
                 d_out,
                 d_out_hid=32,
                 dropout=0.3,
                 pooling_method="cat",
                 pooling_layer_idx=-1
                 ):
        super(ChebConvNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.out_channels = out_channels
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        self.conv1 = ChebConv(self.in_channels, self.hidden_channel, K=3)
        self.layers = nn.ModuleList()
        for i in range(num_hiddens):
            self.layers.append(ChebConv(self.hidden_channel, self.hidden_channel, K=3))
        self.conv2 = ChebConv(self.hidden_channel, self.out_channels, K=3)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.SiLU()
        self.proj_out = nn.Sequential(
            nn.Linear(self.out_channels, d_out_hid),
            nn.ReLU(),
            # nn.Linear(d_out_hid, d_out_hid),
            # nn.ReLU(),
            nn.Linear(d_out_hid, d_out),
        )
        # if pooling_method != "cls_token":
        #     self.pool = HomogeneousAggregator(
        #         pooling_method,
        #         pooling_layer_idx,
        #         layer_layout,
        #     )
    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        x = self.conv1(x, edge_index)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.drop(self.activation(x))
        x = self.conv2(x, edge_index)
        print(x.shape)
        x = scatter(x, batch, dim=0, reduce='sum')
        x = self.proj_out(x)
        return x

