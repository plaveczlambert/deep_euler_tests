# -*- coding: utf-8 -*-
import os
import glob
import h5py
import numpy as np

def load_from_txt(path_to_store:str, run_id:str)->np.ndarray:
    txt_prefix  = 'bubble_sim_'
    run_ids     = [
        os.path.split(f)[-1][len(txt_prefix):-4]
        for f in glob.glob(os.path.join(path_to_store, '*.txt'))
        ]
    assert run_id in run_ids
    path= os.path.join(path_to_store, txt_prefix+run_id+'.txt')
    arr = np.loadtxt(path, dtype=np.float64, delimiter=',')
    
    arr = np.delete(arr, 4, axis=1) #fifth col is always 1.0 - unnecessary
    return arr

def load_from_hdf(path_to_store:str, run_id:str)->np.ndarray:
    f = h5py.File(path_to_store, 'r')
    #permutation_n = 375 #375
    #print(list(f.keys()))
    data = f[run_id]
    #print("Data shape:\n " +str(data.shape))
    arr = np.zeros(data.shape)
    data.read_direct(arr)
    f.close()
    return arr

def load_from_hdf2(path_to_store:str, run_id:str)->np.ndarray:
    '''Load preprocessed data from hdf'''
    f = h5py.File(path_to_store, 'r')
    #permutation_n = 375 #375
    print(list(f.keys()))
    X1 = f[run_id+str('_X')]
    Y1 = f[run_id+str('_Y')]
    print("Input and output shape:\n " +str(X1.shape) + "\n" + str(Y1.shape))
    X = np.zeros(X1.shape)
    Y = np.zeros(Y1.shape)
    X1.read_direct(X)
    Y1.read_direct(Y)
    f.close()
    return X,Y

def sequential_split(arr):
    '''Data with current x1,x2,x3'''
    x1 = np.copy(arr)[:-1, :-1] #dtprev x1 x2 x3 z... without grad
    x2 = np.copy(arr)[1:,1:4] #x1now x2now x3now
    x = np.hstack((x1, x2)) #dtprev x1 x2 x3 z... x1now x2now x3now
    x[:,0] = np.copy(arr)[1:, 0] #dt x1 x2 x3 z... x1now x2now x3now
    y = np.copy(arr)[1:, 4:]
    return x, y

def sequential_split_no_dt(arr):
    '''Data with current x1,x2,x3 and no dt for fixed step'''
    x1 = np.copy(arr)[:-1, 1:-1] #x1 x2 x3 z... without grad
    #print(x1[0,:])
    x2 = np.copy(arr)[1:,1:4] #x1now x2now x3now
    #print(x2[0,:])
    x = np.hstack((x1, x2)) #x1 x2 x3 z... x1now x2now x3now
    y = np.copy(arr)[1:, 4:]
    return x, y

def sequential_split_with_past_points_no_dt(arr, prev_n = 1):
    '''Data with current x1,x2,x3, previous points and without dt for fixed step'''
    x0 = np.copy(arr)[:-2, 1:-1] #x1 x2 x3 z... without grad previous
    x1 = np.copy(arr)[1:-1, 1:-1] #x1 x2 x3 z... without grad initial
    #print(x1[0,:])
    x2 = np.copy(arr)[2:,1:4] #x1now x2now x3now
    #print(x2[0,:])
    x = np.hstack((x0, x1, x2)) #x1prev x2prev x3prev zprev... x1 x2 x3 z... x1now x2now x3now
    y = np.copy(arr)[2:, 4:]
    return x, y

def sequential_split2(arr):
    '''Data withOUT current x1,x2,x3'''
    x = np.copy(arr)[:-1, :-1] #dt x1 x2 x3 z... without grad
    x[:,0] = np.copy(arr)[1:, 0] #dt x1 x2 x3 z...
    y = np.copy(arr)[1:, 4:]
    return x, y

def timestamp_to_dt(arr):
    dt          = np.zeros((arr[:, 0].shape))
    dt[1:]      = arr[:-1, 0] #tprev
    arr[:, 0]   -= dt #tnow - tprev = dt, 0->0, 1->dt1
    return arr

def generate_data(arr, n, step=1):
    u = 0
    x = timestamp_to_dt(np.copy(arr))
    X,Y = sequential_split(x)
    for i in range(1,n,step):
        for j in range(i):
            x = timestamp_to_dt(np.copy(arr[:][j::i+1]))
            x,y = sequential_split(x)
            X = np.vstack((X,x))
            Y = np.vstack((Y,y))
    print(X.shape)
    print(Y.shape)  
    return X,Y

def euler_truncation_error(arr, output_size): 
    dt = arr[1:,0] - arr[:-1,0]
    X = np.column_stack((arr[1:,0], arr[:-1,:1+output_size])) #t1 t0 x1(0) x2(0) x3(0) z(0)
    dt_m = np.copy(dt)
    for n in range(1,output_size):
        dt_m = np.column_stack((dt_m,dt))
    Y = np.reciprocal(dt_m*dt_m)*(arr[1:,1:output_size+1] - arr[:-1,1:output_size+1] - dt_m*arr[:-1, output_size+1:])
    return X,Y
    