import numpy as np
import matplotlib.pyplot as plt

def plot_loghist(x, bins, label=None):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins, label=label)
    plt.xscale('log')
    return hist