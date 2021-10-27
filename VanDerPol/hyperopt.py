# -*- coding: utf-8 -*-
#Hyperparameter optimization for freeform neural network
import os
import h5py
import optuna
import optuna.visualization
import plotly
from datetime import datetime
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.optim
import torch.jit
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import MLPs

#reproducibility
torch.manual_seed(410)
np.random.seed(410)
random.seed(410)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
#https://pytorch.org/docs/stable/notes/randomness.html
#https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results


torch.set_default_dtype(torch.float64)
torch.set_num_threads(10)
device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

begin_time = datetime.now();

# ----- ----- ----- ----- ----- -----
# Data loading
# ----- ----- ----- ----- ----- -----
f = h5py.File("shared_drive/vdp_data_dt.hdf5", "r")#os.path.join('shared_drive', 'vdp_data_dt.hdf5'), 'r')
keys = list(f.keys())
print(keys)
X = np.empty(f['vdp_X'].shape)
f['vdp_X'].read_direct(X)
Y = np.empty(f['vdp_Y'].shape)
f['vdp_Y'].read_direct(Y)   
f.close()
    
x_trn, x_vld, y_trn, y_vld  = train_test_split(
    X, Y,
    test_size   = .25,
    random_state= 410,
    shuffle     = True
    )
x_vld, x_tst, y_vld, y_tst  = train_test_split(
    x_vld, y_vld,
    test_size   = .40,
    shuffle     = False
    )
    
in_scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
out_scaler  = StandardScaler(with_mean=True, with_std=True, copy=False)
in_scaler.fit(x_trn)
out_scaler.fit(y_trn)
x_trn   = in_scaler.transform(x_trn)
x_vld   = in_scaler.transform(x_vld)
x_tst   = in_scaler.transform(x_tst)
y_trn   = out_scaler.transform(y_trn)
y_vld   = out_scaler.transform(y_vld)
#y_tst_unnormed = np.array(y_tst,copy=True)
y_tst   = out_scaler.transform(y_tst)

trn_set = TensorDataset(torch.tensor(x_trn, dtype=torch.float64), torch.tensor(y_trn, dtype=torch.float64))
vld_set = TensorDataset(torch.tensor(x_vld, dtype=torch.float64), torch.tensor(y_vld, dtype=torch.float64))
trn_ldr = DataLoader(
    trn_set,
    batch_size  = 100,
    shuffle     = True
    )
vld_batch = 100000
vld_ldr = DataLoader(
    vld_set,
    batch_size  = vld_batch,
    shuffle     = False
    )
    
def objective(trial):
    
    hidden_layers = trial.suggest_int( "hidden_layers", 0, 6)
    neurons_per_layer = list()
    for i in range(hidden_layers):
        neurons_per_layer.append(trial.suggest_int( "neurons_per_layer{}".format(i), 10, 200))
    model   = MLPs.VariableMLP(x_trn.shape[-1], y_trn.shape[-1], neurons_per_layer, hidden_layers)
    model = model.to(device)
    
    learning_rate = 3e-4 #trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    loss    = nn.MSELoss()
    optim   = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-8)#, weight_decay=1e-7)

    vld_loss = 0
    vld_loss_best = 1e100
    epochs_since_best = 0
    for num_epoch in range(5000):
        model.train()
        total_loss  = 0
        for batch in trn_ldr:
            x,y = batch
            x   = x.to(device)
            y   = y.to(device)
            optim.zero_grad()
            out     = model(x)
            trn_loss= loss(out, y)
            trn_loss.backward()
            optim.step()
            total_loss  += trn_loss.item() * len(x)
        total_loss  /= len(trn_ldr.dataset)
        model.eval()
        vld_loss    = 0
        for batch in vld_ldr:
            x,y = batch
            x   = x.to(device)
            y   = y.to(device)
            out = model(x)
            vld_loss += loss(out, y).item() * len(x)
        vld_loss    /= len(vld_ldr.dataset)        
        
        #early_stop
        if vld_loss < vld_loss_best:
            vld_loss_best = vld_loss
            trial.report(vld_loss, num_epoch)
            epochs_since_best = 0
            if trial.should_prune():
                raise optuna.TrialPruned()
        else:
            epochs_since_best += 1
            if epochs_since_best == 50:
                trial.set_user_attr("best_epoch", num_epoch-epochs_since_best)
                return vld_loss_best
            
    print(5000)
    trial.set_user_attr("best_epoch", num_epoch-epochs_since_best)
    trial.set_user_attr("Ran_out_of_epochs", True)
    return vld_loss_best


study_name = "nn_A"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name  = study_name,
    storage     = storage_name,
    sampler     = optuna.samplers.TPESampler(seed=410), 
    pruner      = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10, n_min_trials=5),
    load_if_exists = True,
    direction   = "minimize"
    )
study.optimize(objective, n_trials=500)
end_time = datetime.now()

print("Best value:")
print(study.best_value)
print("params:")
print(study.best_params)

print("_____________")
duration = end_time - begin_time
print("Duration: " + str(duration))

fig = optuna.visualization.plot_optimization_history(study)
fig.update_yaxes(type='log')
fig.write_html(str(study_name)+"_optim_hist.html", include_plotlyjs="cdn")
fig = optuna.visualization.plot_param_importances(study)
fig.write_html(str(study_name)+"_param_importances.html", include_plotlyjs="cdn")
fig = optuna.visualization.plot_intermediate_values(study)
fig.update_yaxes(type='log')
fig.write_html(str(study_name)+"_interm_values.html", include_plotlyjs="cdn")

fig = optuna.visualization.plot_contour(study, params=["hidden_layers", "neurons_per_layer0"])
fig.write_html(str(study_name)+"_cont0.html", include_plotlyjs="cdn")
fig = optuna.visualization.plot_contour(study, params=["hidden_layers", "neurons_per_layer1"])
fig.write_html(str(study_name)+"_cont1.html", include_plotlyjs="cdn")
fig = optuna.visualization.plot_contour(study, params=["hidden_layers", "neurons_per_layer2"])
fig.write_html(str(study_name)+"_cont2.html", include_plotlyjs="cdn")
'''fig = optuna.visualization.plot_contour(study, params=["neurons_per_layer0", "neurons_per_layer1"])
fig.write_html(str(study_name)+"_cont01.html", include_plotlyjs="cdn")
fig = optuna.visualization.plot_contour(study, params=["neurons_per_layer1", "neurons_per_layer2"])
fig.write_html(str(study_name)+"_cont12.html", include_plotlyjs="cdn")'''

print("Plots saved")
