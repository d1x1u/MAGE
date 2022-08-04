# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from torch_geometric.data import Data

from .backbone.cnn import MLP
from .backbone.gnn import GCN, SAGE, GAT, JKNet


class Gnn(torch.nn.Module):
    r"""A multi-layer perception (MLP) mixed with graph neural network model.

    Args:
        channel_list (List[int]): List of input, intermediate and output channels.
            :obj:`len(channel_list) - 1` denotes the number of layers of the MLP.
        num_classes (int): The number of categories in the dataset.
        num_layers (int, optional): The number of GCN layers. (default: :obj:`2`)
    """
    def __init__(self, args, num_features, num_classes: int):
        super().__init__()

        mode = args.mode
        print(args)

        if mode == 'GCN':
            # self.GNN = GCN(in_channels=num_features, hidden_channels=64,
            #                out_channels=num_classes, num_layers=2, dropout=0.5)
            self.GNN = GCN(in_channels=num_features, hidden_channels=64,
                             out_channels=num_classes, num_layers=args.num_layers, dropout=0.5)
        elif mode == 'SAGE':
            self.GNN = SAGE(in_channels=num_features, hidden_channels=64,
                            out_channels=num_classes, num_layers=3, dropout=0.5)
        elif mode == 'GAT':
            self.GNN = GAT(in_channels=num_features, hidden_channels=args.hidden_channels,
                           out_channels=num_classes, num_layers=args.num_layers, heads=args.heads, 
                           dropout=args.dropout, att_dropout=args.att_dropout)
        elif mode == 'JKNet':
            self.GNN = JKNet(in_channels=num_features, hidden_channels=64,
                             out_channels=num_classes, num_layers=2, dropout=0.5, mode='cat')
        elif mode == 'MLP':
            # self.GNN = MLP(in_channels=num_features, hidden_channels=args.hidden_channels, 
            #                 out_channels=num_classes, num_layers=args.num_layers, dropout=0.5)
            self.GNN = MLP(in_channels=num_features, hidden_channels=args.mlp_hidden_channels,
                           out_channels=num_classes, num_layers=args.mlp_num_layers, dropout=args.mlp_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.GNN.reset_parameters()

    def forward(self, data: Data) -> Tensor:
        out = self.GNN(data.x, data.edge_index)
        # out = self.GNN(data.x)
        return out
