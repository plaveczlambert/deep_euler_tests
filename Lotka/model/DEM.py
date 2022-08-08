import numpy as np
from scipy.integrate import ode
from scipy.integrate._ivp.common import warn_extraneous
from scipy.integrate._ivp.base import OdeSolver, DenseOutput

from utils.scalers import StandardScaler

import torch.jit

#torch.set_default_dtype(torch.float64)

class DeepEuler(OdeSolver):
    MODE_GENERAL = 2
    MODE_AUTONOMOUS = 0
    MODE_ABSOLUTE_TIMES = 1
    
    def __init__(self, fun, t0, y0, t_bound, h, traced_model_path, scaler_path="", mode=MODE_GENERAL, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=False)
        
        self.h = h
        self.mode = mode

        self.model = torch.jit.load(traced_model_path)
        self.model.eval()
        if scaler_path:
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
        
        dydt = self.fun(t, y)
        
        if self.mode==DeepEuler.MODE_AUTONOMOUS:
            y_in = np.concatenate(([h], y))
        elif self.mode==DeepEuler.MODE_ABSOLUTE_TIMES:
            y_in = np.concatenate(([t+h], [t], y))
        else:
            y_in = np.concatenate(([t], [h], y))
            
        if self.out_scaler:
            model_out = self.out_scaler.inverse_transform(self.model(torch.tensor(self.in_scaler.transform(y_in))).detach().numpy())
        else:
            model_out = self.model(torch.tensor(y_in)).detach().numpy()
        
        y_new = y + h * dydt + h * h * model_out
        t_new = t + h
        
        self.y_old = y

        self.t = t_new
        self.y = y_new
        
        return (True, None)
        
    def _dense_output_impl(self):
        
        pass