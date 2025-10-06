import pythetatools.toyxp as toyxp
from pythetatools.config import *
from pythetatools.config_visualisation import *
from pythetatools.base_visualisation import show_minor_ticks, plot_histogram, plot_data, plot_stacked_samples
from pythetatools.base_analysis import divide_arrays, poisson_error_bars

from pythetatools.config_samples import sample_to_title, sample_to_nuflav
from pythetatools.file_manager import read_histogram
from pythetatools.config_samples import inter2_to_label, flavour_to_label

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

def plot_data_vs_bestfit_nominalbinning(path_to_data, path_to_asimovbf, outdir_path, save=True):
    """
    Plots data vs expectation observables distribution in the nominal fitting binning

    Parameters
    ----------
    path_to_data : str
        Path to data-file.
    path_to_asimovbf : str
        Path to asimov at best-fit file.
    """
    samples_dict  = toyxp.get_samples_info(path_to_asimovbf)
    asimov_bf     = toyxp.load(path_to_asimovbf, kind="asimov",  breakdown=False)
    data_unbinned = toyxp.load(path_to_data,     kind="data",  samples_dict=samples_dict, tobin=False)
    data_binned   = toyxp.load(path_to_data,     kind="data",  samples_dict=samples_dict, tobin=True)


    for sample_title in samples_dict.keys():
        fig, ax = plt.subplots()
        if sample_title == 'numucc1pi':
            asimov_bf[sample_title].plot(ax, wtag=True, label='Best-fit', color=vermilion)
            dim = '1D'
        else:
            asimov_bf[sample_title].plot(ax, wtag=True, label='Best-fit')
            dim = '2D'
        data_unbinned[sample_title].plot(ax, label='Data')
        show_minor_ticks(ax)
        ax.legend()

        if save:
            fig.savefig(f"{outdir_path}/ToyXP_Data_vs_Bestfit_{dim}_{sample_title}.pdf", bbox_inches='tight')

def plot_data_vs_bestfit_2D_withproj(path_to_data, path_to_asimovbf, outdir_path, save=True):
    """
    Plots data vs expectation observables in 2D with 1D projections

    Parameters
    ----------
    path_to_data : str
        Path to data-file.
    path_to_asimovbf : str
        Path to asimov at best-fit file.
    """
    def negate_labels(x, pos):
        return f"{abs(x):.0f}" 

    samples_dict  = toyxp.get_samples_info(path_to_asimovbf)
    asimov_bf     = toyxp.load(path_to_asimovbf, kind="asimov",  breakdown=False)
    data_unbinned = toyxp.load(path_to_data,     kind="data",  samples_dict=samples_dict, tobin=False)
    data_binned   = toyxp.load(path_to_data,     kind="data",  samples_dict=samples_dict, tobin=True)
    
    
    for sample_title in samples_dict.keys():
        if sample_title == 'numucc1pi':
            continue
        fig = plt.figure(figsize=(8, 6))  # Увеличиваем ширину
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 4.1, 0.2], height_ratios=[1, 3], 
                               wspace=0.05, hspace=0.05)
    
        ax_corner = fig.add_subplot(gs[0, 0])  # Основной график (широкий)
        ax_main = fig.add_subplot(gs[1, 1])  # Основной график (широкий)
        ax_xproj = fig.add_subplot(gs[0, 1], sharex=ax_main)  # Верхняя проекция (узкая)
        ax_yproj = fig.add_subplot(gs[1, 0], sharey=ax_main)  # Левая проекция (узкая)
        ax_cbar = fig.add_subplot(gs[1, 2])  # Ось для цветовой шкалы
    
        #Main
        im = asimov_bf[sample_title].plot(ax_main, wtag=False, wtitle=False, show_colorbar=False, label='Best-fit')
        data_unbinned[sample_title].plot(ax_main, wtag=False, wtitle=False, label='Data')
        ax_main.yaxis.set_tick_params(labelleft=False)
        ax_main.set_ylabel('')
        ax_main.tick_params(axis='both', labelsize=22) 
        ax_main.text(0.50, 0.6, CONFIG.tag, fontsize=15, transform=ax_main.transAxes)
        show_minor_ticks(ax_main)
        ax_main.legend(loc='upper right', framealpha=0)
        
        #Colorbar
        fig.colorbar(im, cax=ax_cbar, cmap=rev_afmhot)
    
        #Upper
        asimov_bf[sample_title].project_to_x().plot(ax_xproj, label='Best-fit', wtitle=False, color=vermilion)
        data_binned[sample_title].project_to_x().plot(ax_xproj, kind='data', label='Data', wtitle=False )
        ax_xproj.set_ylabel('')
        ax_xproj.set_xlabel('')
        ax_xproj.xaxis.set_tick_params(labelbottom=False)
        show_minor_ticks(ax_xproj)
        ax_xproj.legend(fontsize=15)
    
        #Left
        asimov_y_proj = asimov_bf[sample_title].project_to_y()
        data_binned_y_proj = data_binned[sample_title].project_to_y()
    
        plot_histogram(ax_yproj, asimov_y_proj.bin_edges[0], asimov_y_proj.z, rotate=True, color=vermilion)# we do not use built-in method because rotation is necessary
        plot_data(ax_yproj, data_binned_y_proj.bin_edges[0], data_binned_y_proj.z, rotate=True)# we do not use built-in method because rotation is necessary
    
        ax_yproj.set_xlabel('')
        ax_yproj.set_ylabel('Angle, [degrees]')
        ax_yproj.set_ylim(0, 180)
        ax_yproj.xaxis.set_major_formatter(ticker.FuncFormatter(negate_labels))
        show_minor_ticks(ax_yproj)
    
        #Text in corner
        x_center = (ax_corner.get_xlim()[0] + ax_corner.get_xlim()[1]) / 2 * 0.58
        y_center = (ax_corner.get_ylim()[0] + ax_corner.get_ylim()[1]) / 2 * 0.6
        ax_corner.text(x_center, y_center, sample_to_title[sample_title], ha='center', va='center', fontsize=20, color='black')
        ax_corner.set_xticks([])
        ax_corner.set_yticks([])
        ax_corner.set_frame_on(False)

        if save:
            fig.savefig(f"{outdir_path}/ToyXP_Data_vs_Bestfit_with_proj_{sample_title}.pdf", bbox_inches='tight')


def plot_data_vs_bestfit_1D(path_to_data, path_to_asimovbf, outdir_path, save=True):
    """
    Plots data vs expectation observables in 2D with 1D projections

    Parameters
    ----------
    path_to_data : str
        Path to data-file.
    path_to_asimovbf : str
        Path to asimov at best-fit file.
    """

    samples_dict  = toyxp.get_samples_info(path_to_asimovbf)
    asimov_bf     = toyxp.load(path_to_asimovbf, kind="asimov",  breakdown=False)
    data_unbinned = toyxp.load(path_to_data,     kind="data",  samples_dict=samples_dict, tobin=False)
    data_binned   = toyxp.load(path_to_data,     kind="data",  samples_dict=samples_dict, tobin=True)

    for sample_title in samples_dict.keys():
        fig, ax = plt.subplots()
        asimov_bf[sample_title].project_to_x().plot(ax, wtag=True, label='Best-fit', wtitle=True, color=vermilion)
        data_binned[sample_title].project_to_x().plot(ax, kind='data', label='Data', wtitle=False )
        show_minor_ticks(ax)
        ax.legend()

        if save:
            fig.savefig(f"{outdir_path}/ToyXP_Data_vs_Bestfit_1D_{sample_title}.pdf", bbox_inches='tight')

def plot_n_histograms(paths_to_asimov, labels, postfix, colors, outdir_path, save=True):
    """
    Plots expectation observables in 1D projections
    for multiple Asimov best-fit files.

    Parameters
    ----------
    paths_to_asimov : list of str
        List of paths to asimov best-fit files.
    """

    # Берём информацию о сэмплах из первого файла (считаем, что одинакова для всех)
    samples_dict = toyxp.get_samples_info(paths_to_asimov[0])

    # Загружаем все asimov best-fit
    asimov_list = [
        toyxp.load(path, kind="asimov", breakdown=False)
        for path in paths_to_asimov
    ]

    for sample_title in samples_dict.keys():
        fig, ax = plt.subplots()

        # рисуем все best-fit гистограммы
        for i, asimov in enumerate(asimov_list, start=0):
            asimov[sample_title].project_to_x().plot(
                ax, wtag=True, wtitle=True, label=labels[i], color=colors[i],
            )

        show_minor_ticks(ax)
        ax.legend()

        if save:
            fig.savefig(
                f"{outdir_path}/ToyXP_Asimov_1D_{sample_title}_{postfix}.pdf",
                bbox_inches="tight"
            )

def get_labels(sample_per_mode, kind):
    labels = []
    if kind=='inter':
        dict_to_labels = inter2_to_label
    elif kind=='flavour':
        dict_to_labels = flavour_to_label
        
    for sample in sample_per_mode.samples:
        mode_name = sample.title[len(sample.sample_title)+1:]
        labels.append(dict_to_labels[mode_name])   
    return labels


def plot_toyxp_distr_breakdown(ax, sample_per_mode, kind, data_sample=None):
    labels = get_labels(sample_per_mode, kind)
    plot_stacked_samples(ax, sample_per_mode.samples, labels=labels)
    if data_sample:
        data_sample.plot(ax, wtag=True, kind='data')
    show_minor_ticks(ax)
    ax.legend(loc='best')


def plot_toyxp_per_int_mode_or_oscchannel(filename_asimov, filename_data, kind, outdir_path, axis='energy', postfix='AsimovA22_postBanff', save=True):
    samples_dict = toyxp.get_samples_info(filename_asimov, sample_titles=CONFIG.sample_titles)
    asimov_dataset = toyxp.load(filename_asimov, kind="asimov", samples_dict=samples_dict, breakdown=True)
    data = toyxp.load(filename_data, kind="data",  samples_dict=samples_dict)
    asimov_1D = toyxp.project_all_samples(asimov_dataset, axis)

    if kind=='inter':
        toy_brokedown= toyxp.merge_for_inter_plotting(asimov_1D)
    else:
        toy_brokedown = toyxp.merge_for_flavour_plotting(asimov_1D)
    data_1D = toyxp.project_all_samples(data, axis)

    for sample_title in asimov_1D.sample_titles:
        sample_per_int_mode = toy_brokedown.filter_by_sample_title(sample_title)
        
        fig, ax = plt.subplots()
        plot_toyxp_distr_breakdown(ax, sample_per_int_mode, kind, data_1D[sample_title])
        if save:
            fig.savefig(f'{outdir_path}/{sample_title}_axis{axis}_per_int_modes{postfix}.pdf', bbox_inches='tight')



def plot_osc_vs_noosc_samples(filepath_asimovbf, filepath_asimovnoosc, filepath_data, outdir_path, axis='energy'):
    samples_dict = toyxp.get_samples_info(filepath_asimovbf, sample_titles=CONFIG.sample_titles)
    
    asimov_dataset_noosc = toyxp.load(filepath_asimovnoosc, kind="asimov", samples_dict=samples_dict)
    asimov_dataset_bf = toyxp.load(filepath_asimovbf, kind="asimov", samples_dict=samples_dict)
    data = toyxp.load(filepath_data, kind="data",  samples_dict=samples_dict)
    
    asimov_noosc_1D = toyxp.project_all_samples(asimov_dataset_noosc, axis)
    asimov_bf_1D = toyxp.project_all_samples(asimov_dataset_bf, axis)
    data_1D = toyxp.project_all_samples(data, axis)
    
    for sample_title in CONFIG.sample_titles:
        fig = plt.figure(figsize=(10, 10))
    
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # The first subplot will be twice as large as the second
        ax1 = fig.add_subplot(gs[0])  
        ax2 = fig.add_subplot(gs[1])  
        
        asimov_noosc_1D[sample_title].plot(ax1, label='No oscillations', ls='--', color=bluish_green)
        asimov_bf_1D[sample_title].plot(ax1, label='With oscillations', color=vermilion)
        data_1D[sample_title].plot(ax1, wtag=True, kind='data', label='Data')
        
        (asimov_noosc_1D[sample_title]/asimov_noosc_1D[sample_title]).plot(ax2, wtitle=False, wtag=False,
                                                label='No oscillations', ls='--', color=bluish_green)
        (asimov_bf_1D[sample_title]/asimov_noosc_1D[sample_title]).plot(ax2, wtitle=False, wtag=False,
                                                label='With oscillations', color=vermilion)
        
        yerr = [divide_arrays(poisson_error_bars(data_1D[sample_title].z, 1-0.6827)[i],
                              asimov_noosc_1D[sample_title].z) for i in range(2)]
        (data_1D[sample_title]/asimov_noosc_1D[sample_title]).plot(ax2, kind='data', wtitle=False, wtag=False,
                                                                   yerr=yerr, label='Data')
        
        
        show_minor_ticks(ax1)
        show_minor_ticks(ax2)
        ax1.legend()
        ax1.set_xticklabels([])  
        ax1.set_xlabel('')  
        if sample_to_nuflav[sample_title] == 'numu':
            ax2.set_ylim(0, 4.5)
        else:
            ax2.set_ylim(0, 8)
        ax2.set_ylabel('Ratio to No Osc.')
        plt.subplots_adjust(hspace=0)
        fig.savefig(f'{outdir_path}/{sample_title}_BF_vs_Data_vs_Nosoc_axis{axis}.pdf', bbox_inches='tight')

