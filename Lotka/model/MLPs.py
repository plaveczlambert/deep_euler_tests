# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, num_in_features, num_out_features, neurons_per_layer):
        super(SimpleMLP, self).__init__()
        self.act    = nn.ELU()
        self.l_in   = nn.Linear(
            in_features = num_in_features,
            out_features= neurons_per_layer
            )
        self.l1   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l2   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l3   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l4   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l5   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l6   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l_out   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= num_out_features
            )
        #weight init
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.xavier_normal_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)
        torch.nn.init.xavier_normal_(self.l3.weight)
        torch.nn.init.zeros_(self.l3.bias)
        torch.nn.init.xavier_normal_(self.l4.weight)
        torch.nn.init.zeros_(self.l4.bias)
        torch.nn.init.xavier_normal_(self.l5.weight)
        torch.nn.init.zeros_(self.l5.bias)
        torch.nn.init.xavier_normal_(self.l6.weight)
        torch.nn.init.zeros_(self.l6.bias)
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)
        
       

    def forward(self, x):
        x   = self.act(self.l_in(x))
        x   = self.act(self.l1(x))
        x   = self.act(self.l2(x))
        x   = self.act(self.l3(x))
        x   = self.act(self.l4(x))
        x   = self.act(self.l5(x))
        x   = self.act(self.l6(x))
        x   = self.act(self.l_out(x))
        return x
