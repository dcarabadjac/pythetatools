import numpy as np
from .base_analysis import sigma_to_CL, marg_mean
from .config_visualisation import *
from .base_visualisation import show_minor_ticks

from math import log10, floor
from decimal import Decimal

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def round_to_3(x):
    if x > 0:
        return format_e(Decimal(round(x, -int(floor(log10(abs(x))))+2)))
    return "<5E-5"


def sinttt_to_sinmumu(x):
    cos2th13 = 0.510
    c = 1/2/cos2th13 #mirror point
    return 4*x*c*(1-x*c)

def get_pvalue(dchi2, dchi2_df, side):
    if side == 'left':
        p0 = np.sum(dchi2 < dchi2_df) / len(dchi2)
    elif side == 'right':
        p0 = np.sum(dchi2 > dchi2_df) / len(dchi2)
    else:
        raise ValueError ("Not allowed value for side")
    return p0

def plot_dchi2_distr(ax, dchi2, dchi2_df, true_hyp, label, left, hypothesis=None, density=False, fill=True):

    xmin = np.min(dchi2)
    xmax = np.max(dchi2)
    colors = {0:vermilion, 1:bluish_green}
    if hypothesis == 'MO':
        colors = {0:darkblue, 1:darkorange}
        xmin = -15
        xmax = 15
    elif hypothesis == 'octant':
        colors = {0:vermilion, 1:bluish_green}
        xmin = -20
        xmax = 30
    elif hypothesis == 'CPC':
        colors = {0:'indigo', 1:'lightseagreen'}
        xmin = 0
        xmax = 30
    elif hypothesis == 'GoF-rate':
        colors = {0:'green'}
    elif hypothesis == 'GoF-rateshape':
        colors = {0:'green'}

    
    counts, bin_edges = np.histogram(dchi2, bins=np.linspace(xmin, xmax, 100), density=density)
    
    ax.hist(dchi2, zorder=0, alpha=1., edgecolor=colors[true_hyp], 
            histtype='step', linewidth=2, label=label, bins=np.linspace(xmin, xmax, 100), density=density)
    bin_idx = np.searchsorted(bin_edges, dchi2_df) - 1
    if left and fill:
        ax.fill_between(bin_edges[:-1], counts, where=(bin_edges[:-1] < dchi2_df), 
                    color=colors[true_hyp], alpha=0.3, zorder=-1, step='post', edgecolor='none')
        x = np.linspace(bin_edges[bin_idx], dchi2_df, 2)
        ax.fill_between(x, counts[bin_idx],
                    color=colors[true_hyp], alpha=0.3, zorder=-1, edgecolor='none')
    elif not left and fill:
        ax.fill_between(bin_edges[:-1], counts, where=(bin_edges[:-1] > dchi2_df), 
            color=colors[true_hyp], alpha=0.3, zorder=-1, step='post', edgecolor='none')
        x = np.linspace(bin_edges[bin_idx+1], dchi2_df, 2)
        ax.fill_between(x, counts[bin_idx],
                    color=colors[true_hyp], alpha=0.3, zorder=-1, edgecolor='none')

    ax.set_ylabel("Number of toys")
    show_minor_ticks(ax)


def get_percentiles(dchi2, sigmas=[1,2,3]):
    levels = [(1-sigma_to_CL(sigma))/2 for sigma in sigmas] + [0.5] + [1 - (1-sigma_to_CL(sigma))/2 for sigma in sigmas] 
    return np.percentile(dchi2, [l * 100 for l in levels])


def marg_cosdelta(grid_x, avnlllh, prof=False):
    assert (avnlllh.shape[2]-1) % 2 == 0, "If delta grid is not divisible by 2, you should reconsider this implementation"
    
    half = avnlllh.shape[2] // 2
    quarter = avnlllh.shape[2] // 4

    results = []
    new_grid = []

    indices = np.arange(0, 0 + quarter + 1).astype(int)
    sym_indices = (2*0 + 2*quarter + 1 - indices).astype(int)
    new_grid = np.sin(grid_x[indices]) [::-1]
    avnllh_marg = marg_mean(avnlllh[:, :, indices], avnlllh[:, :, sym_indices], prof)[:, :, ::-1]
    indices = np.arange(half+1, half + quarter + 1).astype(int)
    sym_indices = (2*half + 2*quarter + 1 - indices).astype(int)
    new_grid = np.concatenate((new_grid, np.sin(grid_x[indices])))
    avnllh_marg = np.concatenate((avnllh_marg, marg_mean(avnlllh[:, :, indices], avnlllh[:, :, sym_indices], prof)), axis=2)  
    return new_grid, avnllh_marg, 'sindelta'