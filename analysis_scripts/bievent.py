from pythetatools.config_visualisation import *
from pythetatools.config_osc_params import osc_param_to_title
from pythetatools.base_visualisation import show_minor_ticks
import pythetatools.toyxp as toyxp
from pythetatools.base_analysis import poisson_error_bars

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from osc_prob.default_parameters import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from tqdm import tqdm
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

#set line colors, markers type for the bi-event plot
markers = ['v',  '_', 'o', 'P']
fill_styles = ['none', 'full', 'none', 'full']
size_markers = [7, 12, 7, 7]
width_markers = [1.5, 4, 1.5, 2]
color_NO = 'dodgerblue'
color_IO = 'darkorange'
colors_orange = [lightorange, midorange, darkorange, 'brown']
colors_blue = [lightblue, midblue, darkblue, 'darkblue'] 

#nuecc1pi sample has a different name (selections) for OA2023 and for post-OA2023 
nuecc1pi = {'OA2023':'nue1RD', 'OA2024':'nuecc1pi', 'OA2024_sensstudy':'nuecc1pi'}
nuecc1pi_320kA = {'OA2023':'nue1RD_320kA', 'OA2024':'nuecc1pi_320kA', 'OA2024_sensstudy':'nuecc1pi_320kA'}

def get_filename(contours_template, var_param, ivaried, ifixed, idiscr, mode):
    param_name_to_suffix = {'delta':'dCP', 'sin223':'th23', 'dm2':'dm2'}
    if mode == 0:
        idelta = ivaried
        ith23 = ifixed
        idm2 = idiscr
    elif mode == 1:
        idelta = ivaried
        ith23 = ifixed
        idm2 =idiscr
    elif mode == 2:
        idelta = ifixed
        ith23 = ivaried
        idm2 = idiscr
    elif mode == 3:
        idelta = idiscr #in fact, MO is different here
        ith23 = ifixed
        idm2 = ivaried
    return contours_template.format(param_name=param_name_to_suffix[var_param], idelta=idelta, ith23=ith23, idm2=idm2) 

def closest(arr, value, smaller):
    if smaller:
        arr_filtered = arr[arr <= value]  
        if arr_filtered.size == 0:
            return None  
        return arr_filtered[np.argmax(arr_filtered)]
    else:
        arr_filtered = arr[arr >= value]  
        if arr_filtered.size == 0:
            return None  
        return arr_filtered[np.argmin(arr_filtered)]

def get_contour_data(contours_template, var_param, ifixed_param_array, ivary_param_array, E_x_min, E_x_max, E_y_min, E_y_max, xaxis_sample_titles, yaxis_sample_titles, mode):
    
    x_NO = [[] for _ in range(len(ifixed_param_array))]
    y_NO = [[] for _ in range(len(ifixed_param_array))]
    x_IO = [[] for _ in range(len(ifixed_param_array))]
    y_IO = [[] for _ in range(len(ifixed_param_array))]
    
    for i, ifixed in enumerate(ifixed_param_array):
        for ivaried in ivary_param_array:
            asimov = toyxp.load(get_filename(contours_template, var_param, ivaried, ifixed, 0, mode), kind="asimov")
            x_NO[i].append(np.sum(np.fromiter((asimov[sample_title].slice(E_x_min, E_x_max).contsum() for sample_title in xaxis_sample_titles), dtype=float)))
            y_NO[i].append(np.sum(np.fromiter((asimov[sample_title].slice(E_y_min, E_y_max).contsum() for sample_title in yaxis_sample_titles), dtype=float)))
    
    for i, ifixed in enumerate(ifixed_param_array):
        for ivaried in ivary_param_array:
            asimov = toyxp.load(get_filename(contours_template, var_param, ivaried, ifixed, 1, mode), kind="asimov")
            x_IO[i].append(np.sum(np.fromiter((asimov[sample_title].slice(E_x_min, E_x_max).contsum() for sample_title in xaxis_sample_titles), dtype=float)))
            y_IO[i].append(np.sum(np.fromiter((asimov[sample_title].slice(E_y_min, E_y_max).contsum() for sample_title in yaxis_sample_titles), dtype=float)))

    x_NO = np.array(x_NO)
    y_NO = np.array(y_NO)
    x_IO = np.array(x_IO)
    y_IO = np.array(y_IO)
    return x_NO, y_NO, x_IO, y_IO


def main_setup(mode, contours_template, CONFIG):
    sin2th23_array_4 = np.linspace(0.45, 0.6, 4)
    ivary_17 = np.arange(0, 17, 1)
    asimov_dummy = toyxp.load(get_filename(contours_template, 'delta', 0, 0, 0, 0), kind="asimov")
    if mode == 0:
        E_x_min, E_x_max = 0, closest(asimov_dummy['nue1R'].bin_edges[0], 3., True)
        E_y_min, E_y_max = 0, closest(asimov_dummy['nuebar1R'].bin_edges[0], 3., True)
        xaxis_sample_titles = ['nue1R', nuecc1pi[CONFIG.oaver]] if not CONFIG.include_320kA else ['nue1R', nuecc1pi[CONFIG.oaver], 'nue1R_320kA', nuecc1pi_320kA[CONFIG.oaver]]
        yaxis_sample_titles = ['nuebar1R'] if not CONFIG.include_320kA else ['nuebar1R', 'nuebar1R_320kA']
        var_param = 'delta'
        fixed_param = 'sin223'
        ifixed_param_array = [0, 1, 2, 3]
        ivary_param_array = ivary_17
        iparam_marker_array = [0, 4, 8, 12]
        fixed_param_array = sin2th23_array_4
        marker_labels = [ r'$\delta_{CP}=\pi$', r'$\delta_{CP}=-\pi/2$',  r'$\delta_{CP}=0$', r'$\delta_{CP}=+\pi/2$' ]
    
    elif mode == 1:
        E_x_min, E_x_max = closest(asimov_dummy['nue1R'].bin_edges[0], 0.55, False), closest(asimov_dummy['nue1R'].bin_edges[0], 3., True)
        E_y_min, E_y_max = 0, closest(asimov_dummy['nue1R'].bin_edges[0], 0.55, True)
        xaxis_sample_titles = ['nue1R', 'nuebar1R', nuecc1pi[CONFIG.oaver]] if not CONFIG.include_320kA else ['nue1R', 'nuebar1R', nuecc1pi[CONFIG.oaver], 'nue1R_320kA', 'nuebar1R_320kA', nuecc1pi_320kA[CONFIG.oaver]]
        yaxis_sample_titles = ['nue1R', 'nuebar1R', nuecc1pi[CONFIG.oaver]] if not CONFIG.include_320kA else ['nue1R', 'nuebar1R', nuecc1pi[CONFIG.oaver], 'nue1R_320kA', 'nuebar1R_320kA', nuecc1pi_320kA[CONFIG.oaver]]
        var_param = 'delta'
        fixed_param = 'sin223'
        ifixed_param_array = [0, 2]
        ivary_param_array = ivary_17
        iparam_marker_array = [0, 4, 8, 12]
        fixed_param_array = sin2th23_array_4
        marker_labels = [ r'$\delta_{CP}=\pi$', r'$\delta_{CP}=-\pi/2$',  r'$\delta_{CP}=0$', r'$\delta_{CP}=+\pi/2$' ]
    
    elif mode == 2:
        E_x_min, E_x_max = 0, closest(asimov_dummy['numu1R'].bin_edges[0], 1.2, True)
        E_y_min, E_y_max = 0, closest(asimov_dummy['nue1R'].bin_edges[0], 3., True)
        xaxis_sample_titles = ['numu1R', 'numubar1R'] if not CONFIG.include_320kA else ['numu1R', 'numubar1R', 'numu1R_320kA', 'numubar1R_320kA']
        yaxis_sample_titles = ['nue1R', 'nuebar1R', nuecc1pi[CONFIG.oaver]]  if not CONFIG.include_320kA else ['nue1R', 'nuebar1R', nuecc1pi[CONFIG.oaver], 'nue1R_320kA', 'nuebar1R_320kA', nuecc1pi_320kA[CONFIG.oaver]]
        var_param = 'sin223'
        fixed_param = 'delta'
        ifixed_param_array = [0, 1, 2, 3]
        ivary_param_array = ivary_17
        iparam_marker_array = [6, 8, 10, 12]
        fixed_param_array =  [ r'$-\pi/2$',  r'$0$', r'$\pi/2$', r'$\pi$' ]
        marker_labels = [r'$\sin^2\theta_{23}=0.45$', r'$\sin^2\theta_{23}=0.50$', r'$\sin^2\theta_{23}=0.55$',r'$\sin^2\theta_{23}=0.6$']

    elif mode == 3:
        E_x_min, E_x_max = closest(asimov_dummy['numu1R'].bin_edges[0], 0.6, False), closest(asimov_dummy['numu1R'].bin_edges[0], 1.2, True)
        E_y_min, E_y_max = 0, closest(asimov_dummy['numu1R'].bin_edges[0], 0.6, True)
        xaxis_sample_titles = ['numu1R', 'numubar1R', 'numucc1pi'] if not CONFIG.include_320kA else ['numu1R', 'numubar1R', 'numucc1pi', 'numu1R_320kA', 'numubar1R_320kA', 'numucc1pi_320kA']
        yaxis_sample_titles = ['numu1R', 'numubar1R', 'numucc1pi'] if not CONFIG.include_320kA else ['numu1R', 'numubar1R', 'numucc1pi', 'numu1R_320kA', 'numubar1R_320kA', 'numucc1pi_320kA']
        var_param = 'dm2'
        fixed_param = 'sin223'
        ifixed_param_array = [0, 1, 2, 3]
        ivary_param_array = ivary_17
        iparam_marker_array = [6, 8, 10, 12]
        fixed_param_array = sin2th23_array_4
        marker_labels = [r'$\Delta m^2_{32}=2.45 \times 10^{-3}$' + r'eV$^2$', r'$\Delta m^2_{32}=2.50 \times 10^{-3}$' + r'eV$^2$',
                         r'$\Delta m^2_{32}=2.55 \times 10^{-3}$' + r'eV$^2$', r'$\Delta m^2_{32}=2.60 \times 10^{-3}$' + r'eV$^2$']
    
    return E_x_min, E_x_max, E_y_min, E_y_max, xaxis_sample_titles, yaxis_sample_titles, var_param, fixed_param, ifixed_param_array, ivary_param_array, iparam_marker_array, fixed_param_array, marker_labels 


def get_syst_hists(asimovs_systvar, xaxis_sample_titles,yaxis_sample_titles, E_x_min, E_x_max, E_y_min, E_y_max, start_entry, nentries):
    nexp_x_systvar = np.zeros(nentries)
    nexp_y_systvar = np.zeros(nentries)
    
    for itoy in tqdm(range(start_entry, start_entry+nentries)):
        nexp_x_systvar[itoy] = sum(asimovs_systvar[f'{sample_title}_{itoy}'].slice(E_x_min, E_x_max).contsum() for sample_title in xaxis_sample_titles)
        nexp_y_systvar[itoy] = sum(asimovs_systvar[f'{sample_title}_{itoy}'].slice(E_y_min, E_y_max).contsum() for sample_title in yaxis_sample_titles)
    return nexp_x_systvar, nexp_y_systvar


import scipy.ndimage

def get_syst_contour(nexp_x_systvar, nexp_y_systvar):
    x = nexp_x_systvar
    y = nexp_y_systvar
    
    # Create a 2D histogram with 50 bins
    bins = 50
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    
    # Apply Gaussian smoothing to interpolate the histogram
    H_smooth = scipy.ndimage.gaussian_filter(H, sigma=2.0)
    
    # Flatten the smoothed histogram and sort in descending order
    H_flat = H_smooth.flatten()
    H_sorted = np.sort(H_flat)[::-1]  # Sort values in descending order
    
    # Compute the cumulative sum of normalized values
    H_cumsum = np.cumsum(H_sorted) / H.sum()
    
    # Determine the threshold value corresponding to 68.3% (1σ)
    threshold_value = H_sorted[np.searchsorted(H_cumsum, 0.683)]
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, 
                        (yedges[:-1] + yedges[1:]) / 2)
    
    return X, Y, H_smooth, threshold_value


def axes_config(ax, mode, oaver):
    if oaver=='OA2023':
        if mode == 0:
            ax.set_xlim(30, 140)
            ax.set_ylim(10, 30)
            ax.set_xlabel(r'$\nu$ mode e-like candid.', fontsize=20, labelpad=10)
            ax.set_ylabel(r'$\bar{\nu}$ mode e-like candid.', fontsize=20)
            log_leg1 = 'lower left'
        elif mode == 1:
            ax.set_xlim(40, 85)
            ax.set_ylim(25, 90)
            ax.set_xlabel(r'e-like candid. with $E_{rec}>550$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'e-like candid. with $E_{rec}<550$ MeV', fontsize=20)
            log_leg1 = 'upper left'
        elif mode == 2:
            ax.set_xlim(270, 360)
            ax.set_ylim(65, 190)
            ax.set_xlabel(r'$\mu-$like candid. with $E_{rec}<1200$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'e-like candid.', fontsize=20)
            log_leg1 = 'upper left'
        elif mode == 3:
            ax.set_ylim(70, 220)
            ax.set_xlim(110, 280)
            ax.set_xlabel(r'$\mu-$like candid. with $600 < E_{rec}< 1200$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'$\mu-$like candid. with $E_{rec}<600$ MeV', fontsize=20)
            log_leg1 = 'lower left'
    elif oaver=='OA2024':
        if mode == 0:
            ax.set_xlim(40, 140)
            ax.set_ylim(10, 32)
            ax.set_xlabel(r'$\nu$ mode e-like candid.', fontsize=20, labelpad=10)
            ax.set_ylabel(r'$\bar{\nu}$ mode e-like candid.', fontsize=20)
            log_leg1 = 'lower left'
        elif mode == 1:
            ax.set_xlim(40, 85)
            ax.set_ylim(25, 90)
            ax.set_xlabel(r'e-like candid. with $E_{rec}>550$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'e-like candid. with $E_{rec}<550$ MeV', fontsize=20)
            log_leg1 = 'upper left'
        elif mode == 2:
            ax.set_xlim(270, 360)
            ax.set_ylim(65, 190)
            ax.set_xlabel(r'$\mu-$like candid. with $E_{rec}<1200$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'e-like candid.', fontsize=20)
            log_leg1 = 'upper left'
        elif mode == 3:
            ax.set_ylim(70, 240)
            ax.set_xlim(110, 280)
            ax.set_xlabel(r'$\mu-$like candid. with $600 < E_{rec}< 1200$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'$\mu-$like candid. with $E_{rec}<600$ MeV', fontsize=20)
            log_leg1 = 'lower left'
    elif oaver=='OA2024_sensstudy':
        if mode == 0:
            ax.set_xlim(100, 350)
            ax.set_ylim(30, 75)
            ax.set_xlabel(r'$\nu$ mode e-like candid.', fontsize=20, labelpad=10)
            ax.set_ylabel(r'$\bar{\nu}$ mode e-like candid.', fontsize=20)
            log_leg1 = 'lower left'
        elif mode == 1:
            ax.set_xlim(40, 85)
            ax.set_ylim(25, 90)
            ax.set_xlabel(r'e-like candid. with $E_{rec}>550$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'e-like candid. with $E_{rec}<550$ MeV', fontsize=20)
            log_leg1 = 'upper left'
        elif mode == 2:
            ax.set_xlim(270, 360)
            ax.set_ylim(65, 190)
            ax.set_xlabel(r'$\mu-$like candid. with $E_{rec}<1200$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'e-like candid.', fontsize=20)
            log_leg1 = 'upper left'
        elif mode == 3:
            ax.set_ylim(70, 240)
            ax.set_xlim(110, 280)
            ax.set_xlabel(r'$\mu-$like candid. with $600 < E_{rec}< 1200$ MeV', fontsize=20, labelpad=10)
            ax.set_ylabel(r'$\mu-$like candid. with $E_{rec}<600$ MeV', fontsize=20)
            log_leg1 = 'lower left'
    
    return log_leg1


def bievent_plot_fromdata(ax, path_to_asimovbf, path_to_data, mode, x_NO, y_NO, x_IO, y_IO, x_NO_marker, y_NO_marker, x_IO_marker, y_IO_marker,
                 X_syst, Y_syst, H_smooth, threshold_value, E_x_min, E_x_max, E_y_min, E_y_max, CONFIG, fixed_param, ifixed_param_array, fixed_param_array, marker_labels, xaxis_sample_titles, yaxis_sample_titles, contours_template, suffix='', save=True):

    samples_dict = toyxp.get_samples_info(path_to_asimovbf)
    asimovbf = toyxp.load(path_to_asimovbf, kind="asimov")
    data = toyxp.load(path_to_data, kind=CONFIG.data_kind,  samples_dict=samples_dict)

    for i, ifixed_param in enumerate(ifixed_param_array):
        if mode > 1:
            x_dense = np.linspace(y_NO[i][0], y_NO[i][-1], 100)
            f = interp1d(y_NO[i], x_NO[i], kind='quadratic')
            ax.plot(f(x_dense), x_dense, ls='-', color=colors_blue[ifixed_param])
        else:
            ax.plot(x_NO[i], y_NO[i], ls='-', color=colors_blue[ifixed_param])
            
    for i, ifixed_param in enumerate(ifixed_param_array):
        if mode > 1:
            x_dense = np.linspace(y_IO[i][0], y_IO[i][-1], 100)
            f = interp1d(y_IO[i], x_IO[i], kind='quadratic')
            ax.plot(f(x_dense), x_dense, ls='--', color=colors_orange[ifixed_param])
        else:
            ax.plot(x_IO[i], y_IO[i], ls='--', color=colors_orange[ifixed_param])
            
    for i, ifixed_param in enumerate(ifixed_param_array):
        for x, y, marker, fill_style, size, width in zip(x_NO_marker[i], y_NO_marker[i],
                                                         markers, fill_styles, size_markers, width_markers):
            ax.plot(x, y, marker=marker, ls='', fillstyle=fill_style, markerfacecolor=color_NO,
                     markeredgecolor=color_NO, markeredgewidth=width, markersize=size)
    
    for i in range(len(ifixed_param_array)):
        for x, y, marker, fill_style, size, width in zip(x_IO_marker[i], y_IO_marker[i],
                                                         markers, fill_styles, size_markers, width_markers):
            ax.plot(x, y, marker=marker, ls='', fillstyle=fill_style, markerfacecolor=color_IO,
                     markeredgecolor=color_IO, markeredgewidth=width, markersize=size)
    
    
    selected_colors = [colors_orange[i] for i in ifixed_param_array]
    cmap = ListedColormap(selected_colors)
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    cb.set_ticklabels([])
    cb.ax.set_position([0.63, 0.11, 0.6, 0.77])  
    
    selected_colors = [colors_blue[i] for i in ifixed_param_array]
    cmap = ListedColormap(selected_colors)
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    n = len(ifixed_param_array)
    cb.set_ticks([1/2/n + i/n for i in range(n)])
    cb.set_ticklabels([f'{fixed_param_array[ifixed_param]}' for ifixed_param in ifixed_param_array]) 
    cb.ax.set_position([0.655, 0.11, 0.6, 0.77]) 

    contour_filled = plt.contourf(X_syst, Y_syst, H_smooth.T, levels=[threshold_value, H_smooth.max()], colors=['grey'], alpha=0.25)
    contour_line = plt.contour(X_syst, Y_syst, H_smooth.T, levels=[threshold_value], colors='grey', linewidths=1.)

    for marker, label, fillstyle, size, width in zip(markers, marker_labels, fill_styles, size_markers, width_markers):
        plt.plot([], [], marker=marker, label=label, ls='', color='black', markeredgewidth=width, markersize=size, fillstyle=fillstyle)
    
    
    loc_legend1 = axes_config(ax, mode, CONFIG.oaver)
    legend = plt.legend(loc=loc_legend1, fontsize=13)
    ax.add_artist(legend)
    
    x = sum(data[sample_title].slice(E_x_min, E_x_max).contsum() for sample_title in xaxis_sample_titles)
    y = sum(data[sample_title].slice(E_y_min, E_y_max).contsum() for sample_title in yaxis_sample_titles)
    y_err = poisson_error_bars([y],  1 - 0.6827)
    x_err = poisson_error_bars([x],  1 - 0.6827)
    data_handle = plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', capsize=3, label='Data', color='black', linewidth=2)
    
    x = sum(asimovbf[sample_title].slice(E_x_min, E_x_max).contsum() for sample_title in xaxis_sample_titles)
    y = sum(asimovbf[sample_title].slice(E_y_min, E_y_max).contsum() for sample_title in yaxis_sample_titles)
    bestfit_handle = plt.plot(x, y, marker='d', ls='', label='Best-fit', color='black')[0]
            
    handles = [bestfit_handle, data_handle, Patch(facecolor='grey', edgecolor='grey', alpha=0.5, label='68.27% syst. err. at best-fit')]
    
    plt.legend(handles=handles, loc='upper right', fontsize=15, framealpha=0)
    ax.set_title(CONFIG.tag, loc='right')
    
    ax.text(0.80, 0.9, "NO", fontsize=25, color=darkblue, transform=ax.transAxes)
    ax.text(0.9,  0.9, "IO", fontsize=25, color=darkorange, transform=ax.transAxes)
    ax.text(0.99, 1.05, osc_param_to_title[fixed_param][0], fontsize=20, color='black', transform=ax.transAxes)
    
    show_minor_ticks(ax)

    if save:
        fig.savefig(f'{outputs_dir}/plots/{dir_ver}/Bievent_plots/Bievent_plot{suffix}_mode{mode}.pdf', bbox_inches='tight')


def plot_bievent_main(path_to_asimovbf, path_to_data, path_to_asimovbfsystvar, contours_template, modes, CONFIG, start_entry=0, nentries=1000, suffix='', save=True):
    for mode in modes:
        E_x_min, E_x_max, E_y_min, E_y_max, xaxis_sample_titles,\
        yaxis_sample_titles, var_param, \
        fixed_param, ifixed_param_array, ivary_param_array, iparam_marker_array, fixed_param_array, marker_labels = main_setup(mode, contours_template, CONFIG)
        
        x_NO, y_NO, x_IO, y_IO = \
        get_contour_data(contours_template, var_param, ifixed_param_array, ivary_param_array, E_x_min, E_x_max, E_y_min, E_y_max, xaxis_sample_titles, yaxis_sample_titles, mode)
        x_NO_marker, y_NO_marker, x_IO_marker, y_IO_marker = \
        get_contour_data(contours_template, var_param, ifixed_param_array, iparam_marker_array, E_x_min, E_x_max, E_y_min, E_y_max, xaxis_sample_titles, yaxis_sample_titles, mode)
        
        asimovs_systvar = toyxp.load_multiple_hists_from_tree(path_to_asimovbfsystvar, set(xaxis_sample_titles+yaxis_sample_titles), start_entry, nentries)
        
        nexp_x_systvar, nexp_y_systvar = get_syst_hists(asimovs_systvar, xaxis_sample_titles,yaxis_sample_titles, E_x_min, E_x_max, E_y_min, E_y_max, start_entry, nentries)
        X_syst, Y_syst, H_smooth, threshold_value = get_syst_contour(nexp_x_systvar, nexp_y_systvar)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bievent_plot_fromdata(ax, path_to_asimovbf, path_to_data, mode, x_NO, y_NO, x_IO, y_IO, 
                              x_NO_marker, y_NO_marker, x_IO_marker, y_IO_marker, X_syst,
                              Y_syst, H_smooth, threshold_value, E_x_min, E_x_max, E_y_min, E_y_max,                           
                              CONFIG=CONFIG, ifixed_param_array=ifixed_param_array,
                              fixed_param=fixed_param, fixed_param_array=fixed_param_array, marker_labels=marker_labels,
                              xaxis_sample_titles=xaxis_sample_titles, yaxis_sample_titles=yaxis_sample_titles, 
                              contours_template=contours_template, suffix=suffix, save=save)
            


    