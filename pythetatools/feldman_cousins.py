import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from .base_analysis import find_parabold_vertex
from collections import defaultdict
from pathlib import Path



def get_critical_values(filename_template, param_name_flattened, true_param_grid_sorted, true_mh ):
    crit_val_central = defaultdict(list)
    levels = []
    
    for true_param in true_param_grid_sorted:
        filename = filename_template.format(param_name_flattened=param_name_flattened, true_param=true_param, mh=true_mh)
        filepath = Path(filename)
        data = np.load(filepath)
        
        for i, level in enumerate(data['level']):
            crit_val_central[level].append(data['Upper'][i])
        
        levels = data['level']
    
    return crit_val_central

def find_minimum_pertoy_forgivenmh(grid_x, AvNLL_pergrid_pertoy, mh, param_name):
    index_x0_central = np.argmin(AvNLL_pergrid_pertoy[:, mh, :], axis=1) 
    avnll_pertoy_central = AvNLL_pergrid_pertoy[np.arange(AvNLL_pergrid_pertoy.shape[0]), mh, index_x0_central]
    x0_central = grid_x[index_x0_central]

    if param_name == 'delta':
        grid_x_extended_left = np.concatenate([[2*grid_x[0] - grid_x[1]], grid_x[:-1]])
        grid_x_extended_right = np.concatenate([grid_x[1:], [2*grid_x[-1] - grid_x[-2]]])

        index_x0_left = np.argmin(AvNLL_pergrid_pertoy[:, mh, :], axis=1) - 1
        index_x0_left = np.where(index_x0_left < 0, 50 + index_x0_left, index_x0_left) 
        avnll_pertoy_left = AvNLL_pergrid_pertoy[np.arange(AvNLL_pergrid_pertoy.shape[0]), mh, index_x0_left]
        x0_left = grid_x_extended_left[index_x0_central]
        
        index_x0_right = np.argmin(AvNLL_pergrid_pertoy[:, mh, :], axis=1) + 1
        index_x0_right = np.where(index_x0_right > 50, index_x0_right-50, index_x0_right) 
        avnll_pertoy_right = AvNLL_pergrid_pertoy[np.arange(AvNLL_pergrid_pertoy.shape[0]), mh, index_x0_right]
        x0_right = grid_x_extended_right[index_x0_central]
    elif param_name == 'sindelta':
        return avnll_pertoy_central
    else:
        index_x0_left = np.argmin(AvNLL_pergrid_pertoy[:, mh, :], axis=1) - 1
        x0_left = grid_x[index_x0_left]
        avnll_pertoy_left = AvNLL_pergrid_pertoy[np.arange(AvNLL_pergrid_pertoy.shape[0]), mh, index_x0_left]
        index_x0_right = np.argmin(AvNLL_pergrid_pertoy[:, mh, :], axis=1) + 1
        x0_right = grid_x[index_x0_right]
        avnll_pertoy_right = AvNLL_pergrid_pertoy[np.arange(AvNLL_pergrid_pertoy.shape[0]), mh, index_x0_right]    
    
    _, result = find_parabold_vertex(x0_left, x0_central, x0_right,
                            avnll_pertoy_left, avnll_pertoy_central, avnll_pertoy_right)
    return result
    
def find_minimum_pertoy(grid_x, AvNLL_pergrid_pertoy, param_name):
    AvNLL_minimum_pertoy_permh = np.vstack([find_minimum_pertoy_forgivenmh(grid_x, AvNLL_pergrid_pertoy, 0, param_name), find_minimum_pertoy_forgivenmh(grid_x, AvNLL_pergrid_pertoy, 1, param_name)]).T
    AvNLL_minimum_pertoy = np.min(AvNLL_minimum_pertoy_permh, axis=1)
    return AvNLL_minimum_pertoy


def llr_distr(grid_x, AvNLL_pergrid_pertoy, true_param_value, true_mh, param_name):
    AvNLL_minimum_pertoy = find_minimum_pertoy(grid_x, AvNLL_pergrid_pertoy, param_name)
    f = interp1d(grid_x, AvNLL_pergrid_pertoy[:, true_mh, :], axis=1, fill_value='extrapolate')
    return 2*(f(true_param_value) - AvNLL_minimum_pertoy)

def get_eff_error(target_eff, ntoys, upper):
    """
    Computes the Wilson score interval for a binomial distribution.

    Returns the binomial probability p for ntoys trials, where target_eff (t) 
    lies at the edge of the mean(P) ± standard deviation(P).

    Formulas:
      - Mean value: mean(P) = p
      - Standard deviation: sd(P) = sqrt[p(1 - p) / ntoys]
      - target_eff is expressed as: t = p ± sqrt[p(1 - p) / ntoys]
      - This leads to the quadratic equation:
        (t - p)^2 = p(1 - p) / ntoys
        (1 + 1/ntoys) * p^2 - (2t + 1/ntoys) * p + t^2 = 0
      - Solving for p in the form: a * p^2 + b * p + c = 0

    Parameters:
    ----------
    target_eff : float
        Target efficiency (t).
    ntoys : int
        Total number of trials (toys).
    upper : int
        If 1, returns the upper bound of the confidence interval.
        If 0, returns the lower bound.

    Returns:
    -------
    float
        The boundary of the confidence interval.
    """
    
    if ntoys <= 0:
        ValueError ("{ntoys} mush be greater than 1")
    
    a = 1.0 + 1.0 / ntoys
    b = -(2.0 * target_eff + 1.0 / ntoys)
    c = target_eff ** 2
    
    delta = b ** 2 - 4.0 * a * c
    if delta < 0:
        return -1
    
    sqrt_delta = np.sqrt(delta)
    if upper == 1:
        return (-b - sqrt_delta) / (2.0 * a)
    else:
        return (-b + sqrt_delta) / (2.0 * a)

def find_crit_val(dchi2, levels):
    ntoys = len(dchi2)
    dchi2_sorted = np.sort(dchi2)
    crit_val_centrals = []
    crit_val_uppers = []
    crit_val_lowers = []
    
    for level in levels:
        dchi2crit_id = int(len(dchi2_sorted)*level)
        crit_val_centrals.append(dchi2_sorted[dchi2crit_id])
        crit_val_uppers.append(dchi2_sorted[int(get_eff_error(dchi2crit_id/ntoys, ntoys, 0)*ntoys)])
        crit_val_lowers.append(dchi2_sorted[int(get_eff_error(dchi2crit_id/ntoys, ntoys, 1)*ntoys)])

    return crit_val_centrals, crit_val_lowers, crit_val_uppers

