# -*- coding: utf-8 -*-
import argparse
import os
import h5py
from datetime import datetime
from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.jit
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import MLPs
from utils.plot_utils import plot_loghist
from utils.scalers import writeStandardScaler
from utils.scalers import writeMinMaxScaler

torch.set_default_dtype(torch.float64)

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument(
    '--batch',
    default = '100',
    type    = int,
    help    = "Batch size. 0 means training set length. Default is 100."
    )
parser.add_argument(
    '--epoch',
    default = '1',
    type    = int,
    help    = "Number of epochs to train. Default is 1."
    )
parser.add_argument(
    '--load_model',
    default = '',
    type    = str,
    help    = "Path to model dict file to load."
    )
parser.add_argument(
    '--name',
    default = '',
    type    = str,
    help    = "Optional name of the model."
    )
parser.add_argument(
    '--start_epoch',
    default = '0',
    type    = int,
    help    = "Epochs of training of the loaded model. Deprecated"
    )
parser.add_argument(
    '--save_path',
    default = 'training/',
    type    = str,
    help    = "Path to save model. Default is 'training'."
    )
parser.add_argument(
    '--monitor',
    default = 0,
    type    = int,
    help    = "0: no monitoring, 1: show plots on end, 2: monitor all along"
    )
parser.add_argument(
    '--print_losses',
    default=0,
    type=int,
    help    = "Print every nth losses. Default is 0 meaning no print. Option monitor=2 overrides this."
    )
parser.add_argument(
    '--save_plots',
    dest = 'save_plots',
    action = 'store_true',
    help = "If set, saves the plots generated after training."
    )
parser.add_argument(
    '--test',
    dest='test',
    action='store_true',
    help    = "If set, no saving takes place."
    )
parser.add_argument(
    '--print_epoch',
    default = 0,
    type    = int,
    help    = "Print epoch number at every nth epoch. Default is zero, meaning no print."
    )
parser.add_argument(
    '--cpu',
    dest='cpu', 
    action='store_true', 
    help= "If set, training is carried out on the cpu."
    )
parser.add_argument(
    '--early_stop',
    dest='early_stop', 
    action='store_true', 
    help= "Enable early stop when no improvement in validation loss has been achieved for 50 epochs."
    )
parser.add_argument(
    '--num_threads',
    default = 0,
    type    = int,
    help    = "Number of cpu threads to be used by pytorch. Default is 0 meaning same as number of cores."
    )
parser.add_argument(
    '--data',
    default = os.path.join('data', 'vdp_data_dt.hdf5'),
    type    = str,
    help    = "Data to be loaded for training. Default is 'data/vdp_data_dt.hdf5'."
    )

parser.set_defaults(
    feature=False, 
    monitor=False, 
    load_model=False, 
    test=False, 
    cpu=False,
    early_stop=False
    )
args    = parser.parse_args()

if args.num_threads:
    torch.set_num_threads(args.num_threads)
    
if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

#device selection logic
device=0
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


begin_time = datetime.now();
time_str = begin_time.strftime("%y%m%d%H%M")
print("Begin: "+ str(time_str))
if not args.test:
    logfile = open('training/' + (args.name+'_' if args.name else '') + time_str + '.log','w')


#check model availability
if args.load_model:
    if not os.path.exists(args.load_model):
        print("File: " +args.load_model+" does not exist. Abort")
        exit()

# ----- ----- ----- ----- ----- -----
# Data loading
# ----- ----- ----- ----- ----- -----
data_path = args.data
f = h5py.File( data_path, 'r')
keys = list(f.keys())
print(keys)
X = np.empty(f['vdp_X'].shape)
f['vdp_X'].read_direct(X)
Y = np.empty(f['vdp_Y'].shape)
f['vdp_Y'].read_direct(Y)   
f.close()

print(X[1,:])
print(Y[1,:])
input_names = ["x1", "x2"]


print("Train data from: '"+ data_path +"'")
if not args.test:
    print("Train data from: " + data_path, file=logfile)

input_length = X.shape[1]
print("Input length: " +str(input_length))

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

# ----- ----- ----- ----- ----- -----
# Data scaling
# ----- ----- ----- ----- ----- -----
out_scaler  = StandardScaler(with_mean=True, with_std=True, copy=False)
in_scaler = StandardScaler(with_mean=True, with_std=True, copy=False) #MinMaxScaler(feature_range=(0, 1), copy=False)
print("Training shape: " +str(x_trn.shape))
in_scaler.fit(x_trn)
out_scaler.fit(y_trn)
x_trn   = in_scaler.transform(x_trn)
x_vld   = in_scaler.transform(x_vld)
x_tst   = in_scaler.transform(x_tst)
y_trn   = out_scaler.transform(y_trn)
y_vld   = out_scaler.transform(y_vld)
y_tst_unnormed = np.array(y_tst,copy=True)
y_tst   = out_scaler.transform(y_tst)


trn_set = TensorDataset(torch.tensor(x_trn, dtype=torch.float64), torch.tensor(y_trn, dtype=torch.float64))
vld_set = TensorDataset(torch.tensor(x_vld, dtype=torch.float64), torch.tensor(y_vld, dtype=torch.float64))
trn_ldr = DataLoader(
    trn_set,
    batch_size  = len(trn_set) if args.batch==0 else args.batch,
    shuffle     = True
    )
vld_batch = 100000
vld_ldr = DataLoader(
    vld_set,
    batch_size  = vld_batch,
    shuffle     = False
    )

start_epoch = 0
# ----- ----- ----- ----- ----- -----
# Model definition
# ----- ----- ----- ----- ----- -----
model   = MLPs.OptimizedMLP(x_trn.shape[-1], y_trn.shape[-1])
#MLPs.SimpleMLP(x_trn.shape[-1], y_trn.shape[-1], 80)

model_checkpoint = 0
if args.load_model:
    model_checkpoint = torch.load(args.load_model)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    start_epoch = model_checkpoint['epoch']
    #model.load_state_dict(torch.load(args.load_model)) #for previous types
    if not args.test:
        print("Loaded model state from: " + str(args.load_model),file=logfile)
        print("Loaded model state from: " + str(args.load_model))
        
# ----- ----- ----- ----- ----- -----
# Training
# ----- ----- ----- ----- ----- -----
model = model.to(device)
        
loss    = nn.MSELoss()
optim   = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)
#base_lr = 1e-4
#max_lr = 5e-5
#scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr, max_lr, step_size_up=192)
if args.load_model:
    optim.load_state_dict(model_checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])

total_loss_arr  = np.zeros(args.epoch)
vld_loss_arr    = np.zeros(args.epoch)
epochs          = np.linspace(start_epoch,start_epoch+args.epoch-1,args.epoch)

if args.monitor==2: 
    plt.ion()
    plt.figure(num="Training and Validation Losses")
    
if not args.test:
    print("Training...",file=logfile)

learned_epoch = 0
vld_loss_best = 1e100
best_model_state_dict = 0
best_optim_state_dict = 0
best_epoch = 0
for num_epoch in range(args.epoch):
    if args.print_epoch and num_epoch % args.print_epoch == 0:
        print(num_epoch+start_epoch)
    model.train()
    total_loss  = 0
    len_dataset = 0
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
    total_loss_arr[num_epoch] = total_loss
    learned_epoch += 1
    model.eval()
    vld_loss    = 0
    for batch in vld_ldr:
        x, y= batch
        x   = x.to(device)
        y   = y.to(device)
        out     = model(x)
        vld_loss += loss(out, y).item() * len(x)
    vld_loss    /= len(vld_ldr.dataset)
    vld_loss_arr[num_epoch] = vld_loss
    
    #scheduler.step(vld_loss)
    
    if args.monitor==2 or (args.print_losses and num_epoch%args.print_losses==0):
        print(total_loss)
        print(vld_loss)
    if not args.test:
        print(total_loss, file=logfile)
        print(vld_loss, file=logfile)
    
    #real-time plotting
    if args.monitor==2:
        plt.cla()
        plt.plot(epochs, total_loss_arr)
        plt.plot(epochs, vld_loss_arr)
        plt.yscale('log')
        if num_epoch!= 0: plt.xlim([start_epoch, start_epoch+num_epoch])
        plt.pause(0.01)
        
    if args.early_stop and vld_loss < vld_loss_best:
        vld_loss_best = vld_loss
        best_model_state_dict = deepcopy(model.state_dict())
        best_optim_state_dict = deepcopy(optim.state_dict())
        best_epoch = num_epoch
    else:
        if num_epoch-best_epoch == 50:
            if not args.test:
                print("Early stopped", file=logfile)
            print("Early stopped")
            break
    
if not args.test:
    print("Training ready, epochs: " + str(start_epoch) + "..." + str(start_epoch+learned_epoch),file=logfile)

end_time = datetime.now()
duration = end_time - begin_time
time_end_str = end_time.strftime("%y%m%d%H%M")
print("Ended at: "+ time_end_str)
print("Duration: " + str(duration))
if not args.test:
    print("Training duration: " + str(duration),file=logfile)

# ----- ----- ----- ----- ----- -----
#Test
# ----- ----- ----- ----- ----- -----
tst_set = TensorDataset(torch.Tensor(x_tst), torch.Tensor(y_tst))
tst_batch = 100000
tst_ldr = DataLoader(
    tst_set,
    batch_size  = tst_batch,
    shuffle     = False
    ) 

test_loss    = 0
for batch in tst_ldr:
    x, y= batch
    x   = x.to(device)
    y   = y.to(device)
    out = model(x)
    test_loss += loss(out, y).item() * len(x)
test_loss    /= len(tst_ldr.dataset)
print('Test loss: ' + str(test_loss))
if not args.test:
    print('Test loss: ' + str(test_loss),file=logfile)
  
out = out_scaler.inverse_transform(model(torch.tensor(x_tst,dtype=torch.float64).to(device)).cpu().detach().numpy())
test_losses = np.abs(out - y_tst_unnormed)
max_loss = np.max(test_losses)
mean_loss = np.mean(test_losses)
print('Max unnormed loss: ' + str(max_loss))
print('Mean unnormed loss: ' + str(mean_loss))
if not args.test:
    print('Max unnormed loss: ' + str(max_loss),file=logfile)
    print('Mean unnormed loss: ' + str(mean_loss),file=logfile)
    

# ----- ----- ----- ----- ----- -----
#Model Save
# ----- ----- ----- ----- ----- -----

traced_model = 0
if not args.test:
    #save scalers
    f = open(args.save_path+'scaler_' + (args.name+'_' if args.name else '') +time_str + '.psca','w') #chosen this extension
    if type(out_scaler) == StandardScaler:
        writeStandardScaler(f, out_scaler)
    else:
        writeMinMaxScaler(f, out_scaler)
        
    if type(in_scaler) == StandardScaler:
        writeStandardScaler(f, in_scaler)
    else:
        writeMinMaxScaler(f, in_scaler)
    f.close()
    print("Saved scalers.",file=logfile)
    print("Saved scalers.")
    
    if args.early_stop:
        torch.save({
            'epoch': start_epoch+best_epoch,
            'model_state_dict': best_model_state_dict,
            #'scheduler_state_dict': best_scheduler_state_dict,
            'optimizer_state_dict': best_optim_state_dict
            },
            args.save_path+'model_' + (args.name+'_' if args.name else '') + 'e' + str(start_epoch+learned_epoch) + '_' + time_str + '.pt')
    else:
        torch.save({
            'epoch': start_epoch+learned_epoch,
            'model_state_dict': model.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            },
            args.save_path+'model_' + (args.name+'_' if args.name else '') + 'e' + str(start_epoch+learned_epoch) + '_' + time_str + '.pt')
    print("Saved model.",file=logfile)
    print("Saved model.")

    #trace model to be used by C/C++
    if args.early_stop:
        model.load_state_dict(best_model_state_dict)
        model.eval()
    traced_model = torch.jit.trace(model.cpu(), torch.randn((1,x_trn.shape[-1])))
    traced_model.save(args.save_path+'traced_model_' + (args.name+'_' if args.name else '') + 'e'+str(start_epoch+learned_epoch) + '_' + time_str + '.pt')
    print("Saved trace model.",file=logfile)
    print("Saved trace model.")

# ----- ----- ----- ----- ----- -----
# Plotting
# ----- ----- ----- ----- ----- -----
 
if args.monitor>0:
    plt.ion()
    plt.show()
plt.plot(epochs[0:learned_epoch], total_loss_arr[0:learned_epoch], label='Total Loss')
plt.plot(epochs[0:learned_epoch], vld_loss_arr[0:learned_epoch], label='Validation Loss')
plt.yscale('log')
plt.title('Loss Diagram')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
if args.monitor>0:
    plt.show()
if not args.test and args.save_plots:
    plt.savefig(args.save_path+"learning_curve_"+ (args.name+'_' if args.name else '') + time_str+".png", transparent=True)

plt.figure(num="Losses")
plt.title("Loss Distribution of Truncation Error")
for i in range(test_losses.shape[1]):
    plot_loghist(test_losses[:,i], 500, label=input_names[i])
plt.legend()
if args.monitor>0:
        plt.show()
if not args.test and args.save_plots:
    plt.savefig(args.save_path+"Loss_distr_"+time_str+".png", transparent=True)
        
plt.figure(num="Losses (Full)")
plt.title("Loss Distribution of Truncation Error(Full)")
plot_loghist(test_losses.flat, 500)
#plt.hist(test_losses.flat, bins=50)
#plt.ylim([0,500])
plt.xscale('log')
if args.monitor>0:
    plt.ioff()
    plt.show()
    
if not args.test and args.save_plots:
    plt.savefig(args.save_path+"Loss_distr_full_"+time_str+".png", transparent=True)

if not args.test:
    logfile.close()
    

