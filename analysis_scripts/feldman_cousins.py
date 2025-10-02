import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import glob
from collections import defaultdict
import shutil
import uproot

from pythetatools import likelihood
from pythetatools.base_visualisation import *
from pythetatools.config_visualisation import *

from pythetatools.file_manager import read_histogram, download
from pythetatools.config_fc import *
from pythetatools.config_osc_params import osc_param_to_title
from pythetatools.base_analysis import sigma_to_CL, find_parabold_vertex, CL_to_chi2critval

suffix = {'delta':'', 'sin223':'sin223'}
levels = [sigma_to_CL(z_score) for z_score in [1, 2, 3]]
levels = levels + [0.9]
lss = ['-', '--', 'dashdot', 'dotted']
levels.sort()
labels = ['68.27%', '90.00%', '95.45%', '99.73%']

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


def plot_dchi2_distr_fromdata(fig, ax, grid_x, dchi2, param_name_x, true_param_val, true_mh, levels, crit_val_centrals, outdir_path=None, save=True, color=None, plot_crit_val=True, alpha=None):
    if color is None:
        color=color_mo[true_mh]
        
    ax.hist(dchi2, bins=20, color=color, zorder=0, alpha=alpha)
    for level, crit_val_central in zip(levels, crit_val_centrals):
        ax.axvline(crit_val_central, color='black', ls=level_to_ls[level], label=level_to_label[level]+r' $\Delta \chi^2_{c}$='+f'{round(crit_val_central, 2)}')

    ax.set_title('True ' +osc_param_to_title[param_name_x][0]+f' = {round(true_param_val, 2)}; True {mo_to_title[true_mh]}', loc='right')
    ax.set_xlabel("$\Delta \chi^2$")
    ax.set_ylabel("Number of toy exp.")

    ax.set_xlim(0, 20)
    show_minor_ticks(ax)
    
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
    ax.legend()

    if save:
        fig.savefig(f'{outdir_path}/FC_dchi2_distr_{param_name_x}_{true_param_val}_truemh{true_mh}.pdf', bbox_inches='tight')


def feldman_cousins(file_pattern, true_param_val, true_mh, levels, outdir_path=None, outdir_files_path=None, plot=True, save_dchi2=True, save_critvals=True, color=None, plot_crit_val=True, alpha=None):
    fig, ax = plt.subplots()
    grid_x, AvNLL_pergrid_pertoy, param_name_x  = likelihood.load_1D_array(file_pattern) 
    dchi2 = llr_distr(grid_x, AvNLL_pergrid_pertoy, true_param_val, true_mh, param_name_x)
    crit_val_centrals, crit_val_lowers, crit_val_uppers = find_crit_val(dchi2, levels)
    if plot:
        plot_dchi2_distr_fromdata(fig, ax, grid_x, dchi2, param_name_x, true_param_val, true_mh, levels, crit_val_centrals, outdir_path=outdir_path, save=save_dchi2, color=color, plot_crit_val=plot_crit_val, alpha=alpha)
    if save_critvals:
        save_critical_values(crit_val_centrals, crit_val_lowers, crit_val_uppers, param_name_x, true_param_val, true_mh, levels, outdir_files_path)

    

def save_critical_values(crit_val_centrals, crit_val_lowers, crit_val_uppers, param_name, true_param_val, true_mh, levels, outdir_files_path):

    #For root file
    data = {
        "true_mh": [true_mh]*len(levels),
        f"true_{param_name}": [true_param_val]*len(levels),
        "level": levels,
        "Central": crit_val_centrals,
        "Upper": crit_val_uppers,
        "Lower": crit_val_lowers
    }

    #For numpy file
    data_for_np = np.zeros(len(levels), dtype=[
        ('true_mh', float),
        (f'true_{param_name}', float),
        ('level', float),
        ('Central', float),
        ('Upper', float),
        ('Lower', float)
    ])

    # Fill the structured array
    data_for_np['true_mh'] = [true_mh] * len(levels)
    data_for_np[f'true_{param_name}'] = [true_param_val] * len(levels)
    data_for_np['level'] = levels
    data_for_np['Central'] = crit_val_centrals
    data_for_np['Upper'] = crit_val_uppers
    data_for_np['Lower'] = crit_val_lowers

    with uproot.recreate(f"{outdir_files_path}/CriticalDchi2_{param_name}_{true_param_val}_truemh{true_mh}.root") as file:
        file["CriticalDchi2"] = data
    np.save(f"{outdir_files_path}/CriticalDchi2_{param_name}_{true_param_val}_truemh{true_mh}.npy", data_for_np)
    #Duplucate +pi to -pi just for easier drawing
    if param_name == 'delta' and true_param_val == 3.14159265359:
        source_file = f"{outdir_files_path}/CriticalDchi2_{param_name}_{true_param_val}_truemh{true_mh}.npy"
        destination_file = f"{outdir_files_path}/CriticalDchi2_{param_name}_-{true_param_val}_truemh{true_mh}.npy"
        shutil.copy(source_file, destination_file)


def plot_crit_val(param_name, true_mh, CONFIG, outdir_path, outdir_files_path, wtag=True, save=True):
    fig, ax = plt.subplots()
    crit_val_central = defaultdict(list)
    crit_val_lower = defaultdict(list)
    crit_val_upper = defaultdict(list)
    levels = []
    for true_param in true_param_grid_sorted[param_name]:
        data = np.load(f"{outdir_files_path}/CriticalDchi2_{param_name}_{true_param}_truemh{true_mh}.npy")
        for i, level in enumerate(data['level']):
            crit_val_central[level].append(data['Central'][i])
            crit_val_lower[level].append(data['Lower'][i])
            crit_val_upper[level].append(data['Upper'][i])
        levels = data['level']

    ax.axhline(1, ls='--', color='grey', linewidth=2)
    ax.axhline(4, ls='--', color='grey', linewidth=2)
    ax.axhline(9, ls='--', color='grey', linewidth=2)
    ax.axhline(CL_to_chi2critval(0.9, dof=1), ls='--', color='grey', linewidth=2)
    
    for level in levels:
        ax.plot(true_param_grid_sorted[param_name], crit_val_central[level], color=critval_level_to_color[true_mh][level], marker='o', linewidth=2, label=level_to_label[level])
        ax.plot(true_param_grid_sorted[param_name], crit_val_lower[level], color=critval_level_to_color[true_mh][level],  linewidth=1)
        ax.plot(true_param_grid_sorted[param_name], crit_val_upper[level], color=critval_level_to_color[true_mh][level],  linewidth=1)
        ax.fill_between(true_param_grid_sorted[param_name], crit_val_lower[level], crit_val_upper[level], hatch='//',
                        facecolor=critval_level_to_color[true_mh][level], edgecolor='white', alpha=0.5)

    ax.set_ylim(0, 11)
    ax.set_xlim(true_param_grid_sorted[param_name][0], true_param_grid_sorted[param_name][-1])
    ax.set_xlabel(osc_param_to_title[param_name][0])
    ax.set_ylabel(r'$\Delta \chi^2_c$')
    
    show_minor_ticks(ax)
    ax.legend(ncol=2, loc='upper center')

    ax.set_title(f'True {mo_to_title[true_mh]}', loc='right')
    ax.set_ylim(0, 15)
    if wtag:
        ax.set_title(CONFIG.tag, loc='left')
    
    if save:
        fig.savefig(f'{outdir_path}/FC_critical_values_vs_truevalue_{param_name}_truemh{true_mh}.pdf', bbox_inches='tight')

