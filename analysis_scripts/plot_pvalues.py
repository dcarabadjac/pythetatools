import pythetatools.likelihood as likelihood
from pythetatools.base_analysis import sigma_to_CL, marg_mean
from pythetatools.base_visualisation import show_minor_ticks
from pythetatools.config_visualisation import *
from pythetatools import config as cfg
from pythetatools.config import outputs_dir
from .feldman_cousins import llr_distr
from pythetatools.base_analysis import get_double_sided_gaussian_zscore, sigma_to_CL, CL_to_chi2critval

from matplotlib import pyplot as plt
from math import log10, floor
from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simps


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

def sinttt_to_sin2223(x):
    return 4*x*(1-x)

def get_pvalue(dchi2, dchi2_df, side):
    if side == 'left':
        p0 = np.sum(dchi2 < dchi2_df) / len(dchi2)
    elif side == 'right':
        p0 = np.sum(dchi2 > dchi2_df) / len(dchi2)
    else:
        raise ValueError ("Not allowed value for side")
    return p0

def plot_dchi2_distr(ax, dchi2, dchi2_df, true_hyp, label, left, hypothesis=None, density=False, fill=True, nbins=100):

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

    
    counts, bin_edges = np.histogram(dchi2, bins=np.linspace(xmin, xmax, nbins), density=density)
    
    ax.hist(dchi2, zorder=0, alpha=1., edgecolor=colors[true_hyp], 
            histtype='step', linewidth=2, label=label, bins=np.linspace(xmin, xmax, nbins), density=density)
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



def plot_MO_dchi2(base_dir_toys_true_no, base_dir_toys_true_io, base_dir_data, true_dcp_str, outdir_path, outdir_files_path, save=True):
    fig, ax = plt.subplots()
    avnllh = [None]*2
    base_dirs = [base_dir_toys_true_no, base_dir_toys_true_io]

    #Read data-fit
    file_pattern = f"{base_dir_data}/marg*.root"
    grid_x, avnllh_df, param_name_x = likelihood.load_1D_array(file_pattern)
    avnllh_df = avnllh_df.reshape(avnllh_df.size//2, 2) #remove additional dimension remained for cont.osc.param with 1 point grid

    #Read toy-fits
    for true_mh in range(2):
        file_pattern = f"{base_dirs[true_mh]}/marg*.root"
        grid_x, AvNLL_pergrid_pertoy, param_name_x  = likelihood.load_1D_array(file_pattern)
        if true_dcp_str == 'Posterior':
            avnllh[true_mh] = AvNLL_pergrid_pertoy.reshape(AvNLL_pergrid_pertoy.size//2, 2)
        else:
            llhood = np.exp(-AvNLL_pergrid_pertoy)
            llhood_marginal = simps(llhood, grid_x, axis=2) #Marginalize over delta cp
            avnllh[true_mh] = -np.log(llhood_marginal)

    dchi2_no = 2*(avnllh[0][:, 0] - avnllh[0][:, 1])
    dchi2_io = 2*(avnllh[1][:, 0] - avnllh[1][:, 1])
    dchi2_df = 2*(avnllh_df[:, 0] - avnllh_df[:, 1])[0]

    plot_dchi2_distr(ax, dchi2_no, dchi2_df, 0, label=f'True {mo_to_title[0]}', left=False, hypothesis='MO')
    plot_dchi2_distr(ax, dchi2_io, dchi2_df, 1, label=f'True {mo_to_title[1]}', left=True,  hypothesis='MO')

    p_io = get_pvalue(dchi2_io, dchi2_df, 'left')
    p_no = get_pvalue(dchi2_no, dchi2_df, 'right')
    
    percentiles_no = get_percentiles(dchi2_no)
    percentiles_io = get_percentiles(dchi2_io)

    if true_dcp_str=='Posterior':
        np.save(f"{outdir_files_path}/MO_percentiles_truemh0_deltaposterior.npy", percentiles_no)
        np.save(f"{outdir_files_path}/MO_percentiles_truemh1_deltaposterior.npy", percentiles_io)
    else:
        np.save(f"{outdir_files_path}/MO_percentiles_truemh0_truedelta{true_dcp_str}.npy", percentiles_no)
        np.save(f"{outdir_files_path}/MO_percentiles_truemh1_truedelta{true_dcp_str}.npy", percentiles_io)
        
    ax.set_yscale('log')
    
    ymin, ymax = ax.get_ylim()
    
    ax.set_xlabel("$\chi^2_{NO} - \chi^2_{IO} $")
    ax.set_xlim(-15, 18)

    ax.text(0.05, 0.9, f'$p_{{0}}(IO)$={round_to_3(p_io)}', transform=ax.transAxes, fontsize=18 )
    ax.text(0.05, 0.8, f'$p_{{0}}(NO)$={round_to_3(p_no)}', transform=ax.transAxes, fontsize=18 )

    if true_dcp_str == 'Posterior':
        pass
        #Do we want to have a title for this case?
        #ax.set_title("Posterior $\delta_{CP}$", loc='right')
    else:
        ax.set_title(f"$\delta_{{CP}}^{{True}}={round(float(true_dcp_str), 2)}$", loc='right')
    
    ax.set_title(cfg.CONFIG.tag, loc='right')
    ax.axvline(dchi2_df, color='black', label=f'Data-fit \n ($\Delta\chi^2 = {np.round(dchi2_df, 2)}$)', ls='--', linewidth=2)
    ax.legend(framealpha=0, loc='upper right', bbox_to_anchor = (0.99, 1), fontsize=18)

    if save:
        fig.savefig(f'{outdir_path}/MO_test_dchi2_distrs_delta{true_dcp_str}.pdf', bbox_inches='tight')



def plot_dchi2_foroctant(base_dir, prefix, prior='s2223', save=True):
    
    file_pattern = f"{base_dir}/marg*.root"
    grid_x, avnllh, param_name = likelihood.load_1D_array(file_pattern)

    #Sensitive to the choice of the grid.
    Llow_1 = 0
    Llow_2 = 36
    Lup_1 = Llow_2
    Lup_2 = 2*Llow_2

    if prior == 's223':
        grid_lower = grid_x[Llow_1:Llow_2] 
        grid_upper = grid_x[Lup_1:Lup_2]
        assert np.allclose(grid_lower, 1-grid_upper[::-1]), "Check the definition of Llow_1, Llow_2 etc"
        xlabel = '$\sin^2\\theta_{23}$'
    elif prior == 's2223':
        grid_lower = sinttt_to_sin2223(grid_x)[Llow_1:Llow_2] 
        grid_upper = sinttt_to_sin2223(grid_x)[Lup_1:Lup_2]
        assert np.allclose(grid_lower, grid_upper[::-1]), "Check the definition of Llow_1, Llow_2 etc"
        xlabel = '$\sin^2 2\\theta_{23}$'

    
   
    llhood = np.exp(-avnllh)
    llhood_marg_mo = simps(llhood, [0, 1], axis=1)[0]
    llhood_marginal_lo = simps(llhood_marg_mo[Llow_1:Llow_2], grid_lower, axis=0)/(grid_lower[-1] - grid_lower[0])
    llhood_marginal_uo = simps(llhood_marg_mo[Lup_1:Lup_2], grid_upper, axis=0)/(grid_upper[-1] - grid_upper[0])
    
    avnllh_octant_df = np.c_[-np.log(llhood_marginal_lo), -np.log(llhood_marginal_uo)]
    fig, ax = plt.subplots()
    
    ax.plot(grid_lower, -2*np.log(llhood_marg_mo)[Llow_1:Llow_2], color=vermilion, label='Lower Octant')
    ax.plot(grid_upper, -2*np.log(llhood_marg_mo)[Lup_1:Lup_2], color=bluish_green, label='Upper Octant')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\chi^2 (\sin^2 2\\theta_{23}, TO)$')
    ax.set_title(cfg.CONFIG.tag, loc='right')
    ax.legend()
    show_minor_ticks(ax)
    
    if save:
        fig.savefig(f'{outputs_dir}/plots/{cfg.CONFIG.dir_ver}/dchi2/dchi2_{prefix}_sinth2223_octant_flat{prior}.pdf', bbox_inches='tight')


def plot_dchi2_vs_sindelta(base_dir, prefix):
    file_pattern = f"{base_dir}/marg*.root"
    grid, avnllh_df, param_name = likelihood.load(file_pattern)
    llh_dcp = likelihood.Loglikelihood(grid, avnllh_df, param_name)
    
    AvNLL_pergrid_pertoy = np.stack([avnllh_df[0], avnllh_df[1]], axis=1)
    
    #Marginalisation on cosdelta
    grid_sindcp, AvNLL_pergrid_pertoy_marg, param_name_x = marg_cosdelta(grid[0], AvNLL_pergrid_pertoy)
    avnllh_df_sindcp = {i: AvNLL_pergrid_pertoy_marg[:, i, :] for i in range(2)}
    llh_sindcp = likelihood.Loglikelihood([grid_sindcp], avnllh_df_sindcp, [param_name_x])
    
    dchi2_data_sindcp = [None]*2
    for mh in [0, 1]:
        dchi2_data_sindcp[mh] = llr_distr(grid_sindcp, AvNLL_pergrid_pertoy_marg, 0, mh, param_name_x)
    
    fig, ax = plt.subplots()
    
    plt.plot(np.sin(llh_dcp.grid[0]), llh_dcp.dchi2[0], color='darkblue')
    plt.plot(np.sin(llh_dcp.grid[0]), llh_dcp.dchi2[1], color='brown')
    
    ax.text(0.05, 0.7, "Dark color - $\Delta \chi^2(\delta_{CP})$", transform=ax.transAxes, fontsize=20)
    ax.text(0.05, 0.6, "Light color - $\Delta \chi^2(\sin \delta_{CP})$", transform=ax.transAxes, fontsize=20)
    
    llh_sindcp.plot(ax, wtag=True)
    ax.set_ylim(-0.5, 30)
    
    fig.savefig(f"{outputs_dir}/plots/{cfg.CONFIG.dir_ver}/dchi2/dchi2_{prefix}_sindelta.pdf", bbox_inches='tight')




def plot_octant_dchi2(base_dir_toys_true_lo_no, base_dir_toys_true_lo_io, 
                      base_dir_toys_true_uo_no, base_dir_toys_true_uo_io, io_over_no_ratio,
                      base_dir_data, octant_pair, prior, outdir_path, outdir_files_path, prof=False, save_files=True, save_pdf=True):

    fig, ax = plt.subplots()
    base_dirs = [[base_dir_toys_true_lo_no, base_dir_toys_true_lo_io], [base_dir_toys_true_uo_no, base_dir_toys_true_uo_io]]

    #Be careful, these values depend on the chosen grid. To be improved
    Llow_1 = 0
    Llow_2 = 36
    Lup_1 = Llow_2
    Lup_2 = 2*Llow_2
    #Read data-fit
    file_pattern = f"{base_dir_data}/marg*.root"
    grid_x, avnllh, param_name_x = likelihood.load_1D_array(file_pattern)

    if prior == 's223':
        grid_lower = grid_x[Llow_1:Llow_2] 
        grid_upper = grid_x[Lup_1:Lup_2]
        assert np.allclose(grid_lower, 1-grid_upper[::-1]), "Check the definition of Llow_1, Llow_2 etc"

    elif prior == 's2223':
        grid_lower = sinttt_to_sin2223(grid_x)[Llow_1:Llow_2] 
        grid_upper = sinttt_to_sin2223(grid_x)[Lup_1:Lup_2]
        assert np.allclose(grid_lower, grid_upper[::-1]), "Check the definition of Llow_1, Llow_2 etc"


    llhood = np.exp(-avnllh)
    llhood_marg_mo = simps(llhood, [0, 1], axis=1)[0]

    if not prof:
        llhood_marginal_lo = simps(llhood_marg_mo[Llow_1:Llow_2], grid_lower, axis=0)/(grid_lower[-1] - grid_lower[0]) 
        llhood_marginal_uo = simps(llhood_marg_mo[Lup_1:Lup_2], grid_upper, axis=0)/(grid_upper[-1] -grid_upper[0]) 
    else:
        llhood_marginal_lo = np.max(llhood_marg_mo[Llow_1:Llow_2], axis=0)
        llhood_marginal_uo = np.max(llhood_marg_mo[Lup_1:Lup_2], axis=0)
  
    avnllh_octant_df = np.c_[-np.log(llhood_marginal_lo), -np.log(llhood_marginal_uo)]
    
    #Read toys
    avnllh = [None]*2
    for true_to in range(2):
        file_pattern = f"{base_dirs[true_to][0]}/m*.root"
        grid_x, AvNLL_pergrid_pertoy_no, param_name_x = likelihood.load_1D_array(file_pattern)
        file_pattern = f"{base_dirs[true_to][1]}/m*.root"
        grid_x, AvNLL_pergrid_pertoy_io, param_name_x = likelihood.load_1D_array(file_pattern)

        N_no = AvNLL_pergrid_pertoy_no.shape[0]/(1+io_over_no_ratio)
        AvNLL_pergrid_pertoy = np.vstack((AvNLL_pergrid_pertoy_no[:int(N_no)], AvNLL_pergrid_pertoy_io[:int(N_no*io_over_no_ratio)]))

        #Different values from above because there is different for FC fits and for data-fit
        Llow_1 = 0
        Llow_2 = 41
        Lup_1 = 40
        Lup_2 = 2*41 - 1

        if prior == 's223':
            grid_lower = grid_x[Llow_1:Llow_2] 
            grid_upper = grid_x[Lup_1:Lup_2]
            assert np.allclose(grid_lower, 1-grid_upper[::-1]), "Check the definition of Llow_1, Llow_2 etc"

        elif prior == 's2223':
            grid_lower = sinttt_to_sin2223(grid_x)[Llow_1:Llow_2] # :38 for mumu works only if total number of grid points is 81
            grid_upper = sinttt_to_sin2223(grid_x)[Lup_1:Lup_2]
            assert np.allclose(grid_lower, grid_upper[::-1]), "Check the definition of Llow_1, Llow_2 etc"

            
        llhood = np.exp(-AvNLL_pergrid_pertoy)
        llhood_marg_mo = simps(llhood, [0, 1], axis=1)
        if not prof:
            llhood_marginal_lo = simps(llhood_marg_mo[:, Llow_1:Llow_2], grid_lower, axis=1)/(grid_lower[-1] - grid_lower[0]) 
            llhood_marginal_uo = simps(llhood_marg_mo[:, Lup_1:Lup_2], grid_upper, axis=1)/(grid_upper[-1] -grid_upper[0]) 
        else:
            llhood_marginal_lo = np.max(llhood_marg_mo[:, Llow_1:Llow_2], axis=1)
            llhood_marginal_uo = np.max(llhood_marg_mo[:, Lup_1:Lup_2], axis=1)
        avnllh[true_to] = np.c_[-np.log(llhood_marginal_lo), -np.log(llhood_marginal_uo)]
    
    dchi2_lo = 2*(avnllh[0][:, 1] - avnllh[0][:, 0]) #Upper - Lower octant
    dchi2_uo = 2*(avnllh[1][:, 1] - avnllh[1][:, 0])
    dchi2_df = 2*(avnllh_octant_df[:, 1]-avnllh_octant_df[:, 0])[0]
    
    plot_dchi2_distr(ax, dchi2_lo, dchi2_df, 0, f"True LO", True, 'octant')
    plot_dchi2_distr(ax, dchi2_uo, dchi2_df, 1, f"True UO", False, 'octant')

    p_lo = get_pvalue(dchi2_lo, dchi2_df, 'left')
    p_uo = get_pvalue(dchi2_uo, dchi2_df, 'right')

    p_lo_corr = []
    for i in range(9000):
        p_lo_corr.append(get_pvalue(dchi2_lo, dchi2_lo[i], 'left'))

    percentiles_uo = get_percentiles(dchi2_uo)
    percentiles_lo = get_percentiles(dchi2_lo)

    if save_files:
        np.save(f"{outdir_files_path}/Octant_percentiles_truesin223{octant_pair[0]}_prof{prof}_flat{prior}.npy", percentiles_lo)
        np.save(f"{outdir_files_path}/Octant_percentiles_truesin223{octant_pair[1]}_prof{prof}_flat{prior}.npy", percentiles_uo)
        np.save(f"{outdir_files_path}/Octant_test_statistics_data_result_prof{prof}_flat{prior}.npy", dchi2_df)
        np.save(f"{outdir_files_path}/plo_forcorrtests_truesin223{octant_pair[0]}_prof{prof}_flat{prior}.npy", p_lo_corr)

    
    ymin, ymax = ax.get_ylim()

    ax.set_ylim(0, ymax * 1.5)
    ax.set_xlim(-18, 10)
    ax.set_xlabel("$\chi^2_{UO} - \chi^2_{LO} $")
    if isinstance(octant_pair[0], float):
        ax.set_title(f"$\sin^2 2\\theta^{{\mathrm{{true}}}}_{{23}}={round(sinttt_to_sin2223(octant_pair[0]), 4)}$", loc='left')
        
    ax.set_title(cfg.CONFIG.tag, loc='right')
    ax.text(0.63, 0.9, f'$p_{{0}}(LO)$={round_to_3(p_lo)}', transform=ax.transAxes, fontsize=18 )
    ax.text(0.63, 0.8, f'$p_{{0}}(UO)$={round_to_3(p_uo)}', transform=ax.transAxes, fontsize=18 )
    
    ax.axvline(dchi2_df, color='black', label=f'Data-fit ($\Delta\chi^2 = {np.round(dchi2_df, 2)}$)', ls='--', linewidth=2)
    ax.legend(framealpha=1, loc='upper left', bbox_to_anchor = (0.0, 1))
    if isinstance(octant_pair[0], float):    
        fig.savefig(f'{outdir_path}/Octant_test_dchi2_distrs_sin2223_{4*octant_pair[0]*(octant_pair[1])}_prof{prof}_flat{prior}.pdf', bbox_inches='tight')
    else:
        fig.savefig(f'{outdir_path}/Octant_test_dchi2_distrs_sin2223_posterior_prof{prof}_flat{prior}.pdf', bbox_inches='tight')


def plot_CPC_dchi2(base_dir_data,
                   true_mh,
                   outdir_path,
                   mode="sindelta", 
                   base_dir_toys_true_pi=None,
                   base_dir_toys_true_0=None,
                   base_dir_toys_true_cpv=None,
                   zero_over_pi_ratio=None,
                   logscale="log",
                   plot_chi2=False,
                   fill=True,
                   save=True):
    """
    Plot CPC Δχ² distributions.

    Parameters
    ----------
    ax : matplotlib axis
    base_dir_data : str
        Path to data directory.
    true_mh : int
        True mass hierarchy.
    mode : str
        "single" (test δ=0 or π) or "compare" (CPC vs CPV).
    icpc : int, optional
        If mode="single": 0 → δ=0, 1 → δ=π.
    base_dir_toys : str, optional
        Toys path for mode="single".
    base_dir_toys_true_pi : str, optional
        Toys with true δ=π (mode="compare").
    base_dir_toys_true_0 : str, optional
        Toys with true δ=0 (mode="compare").
    base_dir_toys_true_cpv : str, optional
        Toys with true CPV (mode="compare").
    zero_over_pi_ratio : float, optional
        Weight for combining 0 and π toys (mode="compare").
    """
    fig, ax = plt.subplots()
    
    # Load data likelihood
    file_pattern = f"{base_dir_data}/marg*.root"
    grid, avnllh_df, param_name = likelihood.load(file_pattern)
    AvNLL_pergrid_pertoy = np.stack([avnllh_df[0], avnllh_df[1]], axis=1)

    if mode == "delta":
        # choose δ to test
        if base_dir_toys_true_0 is not None:
            label_dcp, tested_dcp = "0", 0
            base_dir_toys = base_dir_toys_true_0
        elif base_dir_toys_true_pi is not None:
            label_dcp, tested_dcp = r"\pi", np.pi
            base_dir_toys = base_dir_toys_true_pi
        else:
            raise ValueError("At least one of the variables containing toys should not be None")

        grid_x = grid[0]
        dchi2_data = llr_distr(grid_x,
                                               AvNLL_pergrid_pertoy,
                                               tested_dcp,
                                               true_mh,
                                               "delta")[0]

        # load toys
        pattern = f"{base_dir_toys}/marg*.root"
        grid_x, AvNLL_pergrid_pertoy, param_name_x = likelihood.load_1D_array([pattern])

        dchi2 = llr_distr(grid_x, AvNLL_pergrid_pertoy,
                                          tested_dcp, true_mh, "delta")

        # plot
        plot_dchi2_distr(ax, dchi2, dchi2_data, 0,
                                 f"$\Delta \chi^2(\delta_{{CP}}={label_dcp})$",
                                 False, "CPC", fill=fill)

        p0 = get_pvalue(dchi2, dchi2_data, "right")
        ntoys = AvNLL_pergrid_pertoy.shape[0]
        print(f"p0={p0:.3g}, σ≈{get_double_sided_gaussian_zscore(p0):.2f}")

        ax.axvline(dchi2_data, color="black",
                   label=f"Data: $\Delta \chi^2$={np.round(dchi2_data, 2)}",
                   ls="--", linewidth=2)

    elif mode == "sindelta":
        if not (base_dir_toys_true_pi and base_dir_toys_true_0 and base_dir_toys_true_cpv):
            raise ValueError('All variables containing toys should be passed for sindelta mode')
        if not zero_over_pi_ratio:
            raise ValueError('zero_over_pi_ratio should be passed for sindelta mode')
        
        # Marginalize data chi2 over cosdelta
        grid_sindcp, AvNLL_marg, param_name_x = marg_cosdelta(grid[0], AvNLL_pergrid_pertoy)
        dchi2_data = llr_distr(grid_sindcp, AvNLL_marg,
                                               0, true_mh, param_name_x)[0]

        # toys for true δ=π and δ=0

        pattern_pi = f"{base_dir_toys_true_pi}/marg*.root"
        pattern_0 = f"{base_dir_toys_true_0}/marg*.root"
        _, AvNLL_pi, _ = likelihood.load_1D_array([pattern_pi])
        _, AvNLL_0, _ = likelihood.load_1D_array([pattern_0])

        N_pi = AvNLL_pi.shape[0] / (1 + zero_over_pi_ratio)
        AvNLL_cpc = np.vstack((AvNLL_pi[:int(N_pi)],
                               AvNLL_0[:int(N_pi*zero_over_pi_ratio)]))
        ntoys = AvNLL_cpc.shape[0]
        
        grid_x, AvNLL_cpc, param_name_x = marg_cosdelta(grid[0], AvNLL_cpc)

        dchi2_cpc = llr_distr(grid_x, AvNLL_cpc, 0, true_mh, param_name_x)
        plot_dchi2_distr(ax, dchi2_cpc, dchi2_data, 0, "True CPC", False, "CPC")

            
        pattern_cpv = f"{base_dir_toys_true_cpv}/merged*.root"
        _, AvNLL_cpv, _ = likelihood.load_1D_array([pattern_cpv])
        grid_x, AvNLL_cpv, param_name_x = marg_cosdelta(grid[0], AvNLL_cpv)
        dchi2_cpv = llr_distr(grid_x, AvNLL_cpv, 0, true_mh, param_name_x)

        plot_dchi2_distr(ax, dchi2_cpv, dchi2_data, 1, "True Posterior", True, "CPC")

        p_cpc = get_pvalue(dchi2_cpc, dchi2_data, "right")
        p_cpv = get_pvalue(dchi2_cpv, dchi2_data, "left")
        print(f"nsigmas CPC={get_double_sided_gaussian_zscore(p_cpc):.2f}, "
              f"CPV={get_double_sided_gaussian_zscore(p_cpv):.2f}")

        ax.axvline(dchi2_data, color="black",
                   label=f"Data: $\Delta \chi^2$={np.round(dchi2_data, 2)}",
                   ls="--", linewidth=2)
        ax.text(0.70, 0.65, f"$p_0(CPC)$={round_to_3(p_cpc)}", transform=ax.transAxes, fontsize=15)
        ax.text(0.70, 0.58, f"$p_0(Post)$={round_to_3(p_cpv)}", transform=ax.transAxes, fontsize=15)

    if plot_chi2:
        chi2_samples = np.random.chisquare(1, ntoys)
        ax.hist(chi2_samples, bins=np.linspace(0, 30, 100), density=False, color=darkblue, zorder=0, label='$\chi^2$ distribution', 
                    fill=False, histtype='step', linewidth=2)

    # common formatting
    postfix_chi2 = {False:'', True:'_wchi2'}
    ax.set_yscale(logscale)
    ax.set_xlabel(r"$\Delta \chi^2$")
    ax.set_title(f"True {mo_to_title[true_mh]}", loc="left")
    ax.set_title(cfg.CONFIG.tag, loc='right')
    ax.legend(framealpha=1, loc='upper right', fontsize=16, ncols=1)
    ax.set_xlim(0, 20)
    if save:
        fig.savefig(f'{outdir_path}/CPC_test_dchi2_distrs_truemh{true_mh}_mode{mode}{postfix_chi2[plot_chi2]}.pdf', bbox_inches='tight')


def plot_GOF_dchi2(base_dir_toys, base_dir_data, rate_only, outdir_path, save=True):
    #Load toys
    file_pattern = f"{base_dir_toys}/marg*.root"
    grid_x, avnllh, param_name = likelihood.load_1D_array(file_pattern)
    avnllh = avnllh.reshape(avnllh.size//2, 2) 
    llhood = np.exp(-avnllh)
    llhood_marginal = simps(llhood, [0, 1], axis=1) #Marginalize over MO
    avnllh = -np.log(llhood_marginal)

    #Load data value
    file_pattern = f"{base_dir_data}/marg*.root"
    grid_x, avnllh_df, param_name = likelihood.load_1D_array(file_pattern)
    avnllh_df = avnllh_df.reshape(avnllh_df.size//2, 2) 
    llhood = np.exp(-avnllh_df)
    llhood_marginal = simps(llhood, [0, 1], axis=1) #Marginalize over MO
    avnllh_df = -np.log(llhood_marginal)

    fig, ax = plt.subplots()
    
    chi2 = 2*avnllh
    chi2_data = 2*avnllh_df[0]
    
    plot_dchi2_distr(ax, chi2, chi2_data, 0, "Expected distribution", left=False, hypothesis='GoF-rate')
    ax.axvline(chi2_data, color='black', 
                   label=f'Data result', ls='--', linewidth=2)

    if rate_only:
        ax.set_xlabel('$\chi^2_{rate}$')
        postfix = 'rateonly'
    else:
        ax.set_xlabel('$\chi^2_{rate+shape}$')
        postfix = 'rateshape'
        
    p0 = get_pvalue(avnllh, avnllh_df, side='right')
    ax.text(0.75, 0.73, f'$p_{{0}}$={round_to_3(p0)}', transform=ax.transAxes, fontsize=20 )
    
    ax.legend()
    if save:
        fig.savefig(f'{outdir_path}/GOF_test_chi2_distr_{postfix}.pdf', bbox_inches='tight')
    
