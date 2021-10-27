# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, num_in_features:int, num_out_features:int, neurons_per_layer:int):
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
        x   = self.l_out(x)
        return x
    
class VariableMLP(nn.Module):
    def __init__(self, num_in_features:int, num_out_features:int, neurons_per_layer:list, hidden_layers:int):
        super(VariableMLP, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.act    = nn.ELU()
        self.l_in   = nn.Linear(
            in_features = num_in_features,
            out_features= neurons_per_layer[0] if hidden_layers > 0 else num_out_features
            )
        for i in range(1, hidden_layers):
            layer = nn.Linear(
                in_features = neurons_per_layer[i-1],
                out_features= neurons_per_layer[i]
            )
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            setattr(self, "l"+str(i), layer)
            
        if hidden_layers > 0:
            self.l_out   = nn.Linear(
                in_features = neurons_per_layer[hidden_layers-1],
                out_features= num_out_features
            )
            torch.nn.init.xavier_normal_(self.l_out.weight)
            torch.nn.init.zeros_(self.l_out.bias)
        
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        
       

    def forward(self, x):
        x   = self.act(self.l_in(x))
        for i in range(1,self.hidden_layers):
            x = self.act(getattr(self, "l"+str(i)).__call__(x))
        if self.hidden_layers > 0:
            x   = self.l_out(x)
        return x
        
class VariableBoxMLP(nn.Module):
    def __init__(self, num_in_features:int, num_out_features:int, neurons_per_layer:int, hidden_layers:int):
        super(VariableBoxMLP, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.act    = nn.ELU()
        self.l_in   = nn.Linear(
            in_features = num_in_features,
            out_features= neurons_per_layer
            )
        for i in range(0, hidden_layers):
            layer = nn.Linear(
                in_features = neurons_per_layer,
                out_features= neurons_per_layer
            )
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            setattr(self, "l"+str(i), layer)
            
        self.l_out   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= num_out_features
            )
        #weight init
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)
        
       

    def forward(self, x):
        x   = self.act(self.l_in(x))
        for i in range(self.hidden_layers):
            x = self.act(getattr(self, "l"+str(i)).__call__(x))
        x   = self.l_out(x)
        return x
    
class OptimizedMLP(nn.Module):
    def __init__(self, num_in_features:int, num_out_features:int):
        super(OptimizedMLP, self).__init__()
        
        self.act    = nn.ELU()
        
        self.l_in   = nn.Linear(
            in_features = num_in_features,
            out_features= 107
            )
        self.l1   = nn.Linear(
            in_features = 107,
            out_features= 179
            )
        self.l2   = nn.Linear(
            in_features = 179,
            out_features= 179
            )
        self.l3   = nn.Linear(
            in_features = 179,
            out_features= 184
            )
        self.l4   = nn.Linear(
            in_features = 184,
            out_features= 115
            )
        self.l_out   = nn.Linear(
            in_features = 115,
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
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)
        
       

    def forward(self, x):
        x   = self.act(self.l_in(x))
        x   = self.act(self.l1(x))
        x   = self.act(self.l2(x))
        x   = self.act(self.l3(x))
        x   = self.act(self.l4(x))
        x   = self.l_out(x)
        return x    

