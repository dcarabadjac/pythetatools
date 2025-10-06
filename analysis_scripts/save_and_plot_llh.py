from pythetatools.config_osc_params import *
import pythetatools.likelihood as likelihood

from pythetatools.config_fc import true_param_grid_sorted
from pythetatools.config_osc_params import osc_param_name_to_xlabel
from pythetatools.base_visualisation import show_minor_ticks
from pythetatools.file_manager import read_cont
from pythetatools.feldman_cousins import get_critical_values

import os, csv
import numpy as np
import seaborn as sns
import uproot
import ROOT
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator, NullFormatter
import pandas as pd

import os
import subprocess
import numpy as np
import ROOT 
from matplotlib import pyplot as plt


mo_to_suffix = {0:'', 1:'_IH'}
mode_to_prefix = {0:'', 1:'statonly'}
RC_to_suffix = {True: '', False: f'_woRC'}


def save_avnll_hist(llh, outdir):
    """
    Save the average negative log-likelihood (−2 ln L) as a ROOT histogram.

    This function creates ROOT histograms (TH1D or TH2D depending on the
    dimensionality of the scan grid) from the log-likelihood values stored
    in a `pythetatools.likelihood.Loglikelihood` object. The resulting files
    are saved as `hist.root` (normal ordering) and `hist_IH.root` (inverted
    ordering) for backward compatibility with existing C++ macros
    (e.g. `Smear.C`).

    The function also writes metadata about the scanned oscillation parameters
    into the output ROOT files:
      - TParameter<int> objects for parameter enumeration values,
      - TObjString objects with parameter names.

    Parameters
    ----------
    llh : pythetatools.likelihood.Loglikelihood
        Log-likelihood object containing the scan grid, parameter names,
        and the average log-likelihood values.
    outdir : str
        Path to the output directory where ROOT files will be written.

    Outputs
    -------
    ROOT files:
        - hist.root   : histogram for normal ordering (MO = 0)
        - hist_IH.root: histogram for inverted ordering (MO = 1)

    Histogram content:
        - For 1D scans: TH1D with x-axis as the oscillation parameter and
          y-axis as −2 ln L.
        - For 2D scans: TH2D with x- and y-axes as oscillation parameters
          and z-axis as −2 ln L.

    """
    class OscParam:
        not_defined = -1
        sin2213 = 0
        sin223 = 1
        sin223_bar = 2
        deltaCP = 3
        dm2 = 4
        dm2_bar = 5
        sin213 = 6
        OscParamCount = 7

    # Create TObjString for grid parameter name
    OscParamName = {0:"sin2213", 3:"delta", 4:"dm2", 1:"sin223", 6:"sin213"}

    def get_enum_value(param_name):
        mapping = {
            "not_defined": OscParam.not_defined,
            "sin2213": OscParam.sin2213,
            "delta": OscParam.deltaCP,
            "dm2": OscParam.dm2,
            "sin223": OscParam.sin223,
            "sin213": OscParam.sin213,
        }
        return mapping[param_name]
    
    mo_to_suffix = {0: '', 1: '_IH'}
    idx_to_letter = {0:'x', 1:'y'}
    for mo in llh.dchi2.keys():
        grid = llh.grid  # Grid of bin centers
        avnllh = 2 * llh.avnllh[mo]  # Store -2lnL in histogram
    
        if len(grid) == 1:  # 1D Case
            x_bin_centers = grid[0]
            x_bin_edges = np.zeros(len(x_bin_centers) + 1)
            x_bin_edges[0] = x_bin_centers[0] - (x_bin_centers[1] - x_bin_centers[0]) / 2
            x_bin_edges[1:-1] = (x_bin_centers[:-1] + x_bin_centers[1:]) / 2
            x_bin_edges[-1] = x_bin_centers[-1] + (x_bin_centers[-1] - x_bin_centers[-2]) / 2
    
            hist = ROOT.TH1D(f"cont", "cont", len(x_bin_edges) - 1, x_bin_edges)
            hist.GetXaxis().SetTitle(osc_param_to_title[llh.param_name[0]][mo])
            hist.GetYaxis().SetTitle(r'$-2 \ln{L}$')
    
            for i in range(len(avnllh)):
                hist.SetBinContent(i + 1, avnllh[i])
    
        elif len(grid) == 2:  # 2D Case
            x_bin_centers, y_bin_centers = grid
            x_bin_edges = np.zeros(len(x_bin_centers) + 1)
            y_bin_edges = np.zeros(len(y_bin_centers) + 1)
    
            x_bin_edges[0] = x_bin_centers[0] - (x_bin_centers[1] - x_bin_centers[0]) / 2
            x_bin_edges[1:-1] = (x_bin_centers[:-1] + x_bin_centers[1:]) / 2
            x_bin_edges[-1] = x_bin_centers[-1] + (x_bin_centers[-1] - x_bin_centers[-2]) / 2
    
            y_bin_edges[0] = y_bin_centers[0] - (y_bin_centers[1] - y_bin_centers[0]) / 2
            y_bin_edges[1:-1] = (y_bin_centers[:-1] + y_bin_centers[1:]) / 2
            y_bin_edges[-1] = y_bin_centers[-1] + (y_bin_centers[-1] - y_bin_centers[-2]) / 2
    
            hist = ROOT.TH2D(f"cont", "cont",
                             len(x_bin_edges) - 1, x_bin_edges,
                             len(y_bin_edges) - 1, y_bin_edges)
    
            hist.GetXaxis().SetTitle(osc_param_to_title[llh.param_name[0]][mo])
            hist.GetYaxis().SetTitle(osc_param_to_title[llh.param_name[1]][mo])
            hist.GetZaxis().SetTitle(r'$-2 \ln{L}$')
    
            for i in range(len(x_bin_centers)):
                for j in range(len(y_bin_centers)):
                    hist.SetBinContent(i + 1, j + 1, avnllh[i, j])
    
        filename = f"{outdir}/hist{mo_to_suffix[mo]}.root"
        output_file = ROOT.TFile(filename, "RECREATE")
        hist.Write()
    
        gridParams = [get_enum_value(param) for param in llh.param_name]
        for idx, gridParam in enumerate(gridParams):
            paramName = f"{idx_to_letter[idx]}Param"
            pGridParam = ROOT.TParameter(int)(paramName, gridParam)
            pGridParam.Write()
    
            gridParamName = ROOT.TObjString(OscParamName[gridParam])
            gridParamName.Write(f"{idx_to_letter[idx]}ParamName")
    
        print(f'Objects written in {filename}')
        output_file.Close()

    print("Histograms saved to ROOT files.")


def smear_contour(indir, smear_factor=3.3e-5, compile=False):
    """
    Apply smearing to ROOT histograms using the external `Smear` macro.

    This function optionally compiles the `Smear.C` macro into an executable,
    then runs it on ROOT files located in the given directory. It expects
    files named:
        - hist.root
        - hist_IH.root

    For each input file, a smeared version is created with a suffix containing
    the smearing factor (formatted in scientific notation).

    Parameters
    ----------
    indir : str
        Path to the directory containing the input ROOT files.
    smear_factor : float, optional
        Smearing factor to apply (default: 3.3e-5).
    compile : bool, optional
        If True, recompile the Smear macro before running (default: False).

    Outputs
    -------
    Creates new ROOT files in `indir` with names:
        hist_smeared_<smear_factor>.root
        hist_IH_smeared_<smear_factor>.root

    Raises
    ------
    subprocess.CalledProcessError
        If the compilation or execution of the Smear macro fails.
    FileNotFoundError
        If expected input ROOT files are not found in `indir`.
    """
    if compile:
        macro_path = "../pythetatools/macros/Smear.C"
        build_path = "../build/Smear"
        cmd = ["g++", macro_path, "-o", build_path]
        cmd += subprocess.getoutput("root-config --cflags --libs").split()
        subprocess.run(cmd, check=True)

    for suf in ['', '_IH']:
        input_file = os.path.join(indir, f"hist{suf}.root")
        smear_str = f"{smear_factor:.1e}"  # красиво отформатированная запись
        output_file = os.path.join(indir, f"hist{suf}_smeared_{smear_str}.root")

        subprocess.run(["../build/Smear", input_file, output_file, str(smear_factor)], check=True)
        print(f"Saved {output_file}")


def plot_dchi2(indir, outdir_path, outdir_files_path, mode, prefix='', mo='both', FC_filetemplate=None, 
               smeared=False, smear_factor=None, save=False, input_type='cont', filename=None, param=None,
               wRC=True, plot_surface=False, save_root_file=True): 

    smeared_postfix = {False: '', True: f'_smeared_{smear_factor}'}

    #Read from ROOT file.
    avnllh = {}
    if input_type=='cont':
        grid, avnllh, param_names = likelihood.load_from_cont(indir, smeared, smear_factor)
    else:
        grid, avnllh, param_names = likelihood.load_from_SA(indir, filename, param)

    grid, param_names = likelihood.transform_s2213_to_sin213(grid, param_names)

    #This convention was chosen by T2K. Why?
    if len(param_names) == 2:
        mo_treat = 'conditional'
    else:
        mo_treat = 'joint'
        
    llh = likelihood.Loglikelihood(grid, avnllh, param_names, mo_treat=mo_treat)
    param_names_flattened = "".join(param_names)

    #Check if Feldman-Cousins is used
    if FC_filetemplate is not None:
        crit_val_central_NO = get_critical_values(FC_filetemplate, param_names_flattened, true_param_grid_sorted[param_names_flattened], 0)
        crit_val_central_IO = get_critical_values(FC_filetemplate, param_names_flattened, true_param_grid_sorted[param_names_flattened], 1)
        critical_values=[crit_val_central_NO, crit_val_central_IO]
        x_critical_values = true_param_grid_sorted[param_names_flattened]
        show_const_critical = False
        ifFC = '_wFC_crit_values'
    else:
        x_critical_values = None
        critical_values = None
        show_const_critical = True
        ifFC = ''
        
    if not plot_surface:
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    llh.plot(ax, mo=mo, wtag=True, show_const_critical=show_const_critical, x_critical_values=x_critical_values, critical_values=critical_values, plot_surface=plot_surface)

    #Make the x ranges larger for the sin213 without RC
    if not wRC and 'sin213' in param_names:
        ax.set_xlim(0.001, 0.05)
    
    if save:
        fig.savefig(f"{outdir_path}/dchi2_{prefix}{param_names_flattened}{ifFC}{mode_to_prefix[mode]}{smeared_postfix[smeared]}{RC_to_suffix[wRC]}.pdf", bbox_inches='tight')

    if save_root_file:
        output_filename = f"{outdir_files_path}/contour_{param_names_flattened}_{prefix}{ifFC}{mode_to_prefix[mode]}{smeared_postfix[smeared]}{RC_to_suffix[wRC]}.root"
        save_contour(llh, mo, output_filename)
            



def save_contour(llh, mo, output_filename):
    output = ROOT.TFile(output_filename, "RECREATE")
    mo_array = [0, 1] if mo=='both' else [mo]
    
    if llh.ndim() == 2:        
        for imo in mo_array:  
            for level, segments in llh.SA[imo].items():
                for i, segment in enumerate(segments):
                    x_arr = segment[0] 
                    y_arr = segment[1]
                    graph = ROOT.TGraph(len(x_arr), x_arr.astype('float64'), y_arr.astype('float64'))
                    graph_name = f"contour_level{level*10000:.0f}{mo_to_suffix[imo]}_segment{i}"
                    graph.SetName(graph_name)
                    graph.GetXaxis().SetTitle(osc_param_name_to_xlabel[llh.param_name[0]]['both'])
                    graph.GetYaxis().SetTitle(osc_param_name_to_xlabel[llh.param_name[1]]['both'])
                    graph.Write()
    else:
        for imo in mo_array:
            x_arr = llh.grid[0] 
            y_arr = llh.dchi2[imo]
            graph = ROOT.TGraph(len(x_arr), x_arr.astype('float64'), y_arr.astype('float64'))
            graph_name = f"dchi2{mo_to_suffix[imo]}"
            graph.SetName(graph_name)
            graph.GetXaxis().SetTitle(osc_param_name_to_xlabel[llh.param_name[0]]['both'])
            graph.GetYaxis().SetTitle(r"$\Delta \chi^2$")
            graph.Write()

    output.Close()