import numpy as np
from scipy.integrate import ode
from scipy.integrate._ivp.common import warn_extraneous
from scipy.integrate._ivp.base import OdeSolver, DenseOutput

from utils.scalers import StandardScaler

import torch.jit

#torch.set_default_dtype(torch.float64)

class DeepEuler(OdeSolver):

    def __init__(self, fun, t0, y0, t_bound, h, traced_model_path, scaler_path="", absolute_time=False, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=False)
        
        self.h = h
        self.abs_time = absolute_time

        self.model = torch.jit.load(traced_model_path)
        self.model.eval()
        if not absolute_time and scaler_path:
            f = open(scaler_path,'r')
            self.out_scaler = StandardScaler(f)
            self.in_scaler = StandardScaler(f)
            f.close()
        else:
            self.out_scaler = False
        
    def _step_impl(self):
        t = self.t
        y = self.y
        h = self.h
        
        if self.abs_time:
            y_in = np.concatenate(([t+h], [t], y))
        else:
            y_in = np.concatenate(([h], y))
            
        if self.out_scaler:
            model_out = self.out_scaler.inverse_transform(self.model(torch.tensor(self.in_scaler.transform(y_in))).detach().numpy())
        else:
            model_out = self.model(torch.tensor(y_in)).detach().numpy()
        
        y_new = y + h * self.fun(t, y) + h * h * model_out
        t_new = t + h
        
        self.y_old = y

        self.t = t_new
        self.y = y_new
        
        return (True, None)
        
    def _dense_output_impl(self):
        
        pass