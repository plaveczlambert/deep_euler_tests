import numpy as np
from scipy.integrate import ode
from scipy.integrate._ivp.common import warn_extraneous
from scipy.integrate._ivp.base import OdeSolver, DenseOutput

import torch.jit

#torch.set_default_dtype(torch.float64)

class Euler(OdeSolver):

    def __init__(self, fun, t0, y0, t_bound, h, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=True)
        
        self.h = h
        
    def _step_impl(self):
        t = self.t
        y = self.y
        h = self.h
        
        y_new = y + h * self.fun(t, y)
        t_new = t + h
        
        self.y_old = y

        self.t = t_new
        self.y = y_new
        
        return (True, None)
        
    def _dense_output_impl(self):
        
        pass
        