# -*- coding: utf-8 -*-
import os
import glob
import h5py
import numpy as np

def euler_truncation_error(arr, output_size): 
    dt = arr[1:,0] - arr[:-1,0]
    X = np.column_stack((arr[1:,0], arr[:-1,:1+output_size]))
    dt_m = np.copy(dt)
    for n in range(1,output_size):
        dt_m = np.column_stack((dt_m,dt))
    Y = np.reciprocal(dt_m*dt_m)*(arr[1:,1:output_size+1] - arr[:-1,1:output_size+1] - dt_m*arr[:-1, output_size+1:])
    return X,Y
    