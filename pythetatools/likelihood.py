"""
Defines Loglikelihood classes and related functions
"""

from array import array
import uproot
import os
import numpy as np
from . import config as cfg
from .config_visualisation import *
from .config_osc_params import *
from .base_analysis import find_parabold_vertex, sigma_to_CL, CL_to_chi2critval
from .base_visualisation import show_minor_ticks
from .file_manager import read_cont, read_histogram

import sys
import glob
import pandas as pd
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import minimize
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

import ROOT #is needed to write files


def load_from_cont(indir, smeared, smear_factor):
    """
    Load AvNLLtot values from cont histograms stored in hist.root files.

    Parameters
    ----------
    indir : str
        Directory containing the ROOT histogram files.
    smeared : bool
        If True, load smeared likelihood histograms.
    smear_factor : float
        Smearing factor used in the histogram file name.

    Returns
    -------
    grid : list of numpy arrays
        Bin centers for each parameter.
    avnllh : dict
        A dictionary where keys correspond to mass ordering indices (0 for NO, 1 for IO),
        and values are N-dimensional arrays of AvNLL values across the parameter grid.
    param_names : list
        List of parameter names corresponding to grid dimensions.
    """
    
    mo_to_suffix = {0:'', 1:'_IH'}
    smeared_to_postfix = {False: '', True: f'_smeared_{smear_factor}'}
    avnllh = {}
    for mo in [0, 1]:
        bin_edges, double_avnllh, param_names = read_cont(os.path.join(indir, f'hist{mo_to_suffix[mo]}{smeared_to_postfix[smeared]}.root'))
        avnllh[mo] = double_avnllh/2
    grid = [(bin_edges[i][1:]+bin_edges[i][:-1])/2 for i in range(len(bin_edges))]
    return grid, avnllh, param_names

def load_from_SA(indir, filename, param, mo='both'):
    """
    Load 1D dchi2 from SA_* histogram.

    Parameters
    ----------
    indir : str
        Directory containing the ROOT histogram file.
    filename : str
        Name of the ROOT file with sensitivity analysis histograms.
    param : str
        Parameter to load (e.g., 'delta', 'dm2').

    Returns
    -------
    grid : list of numpy arrays
        Bin centers for the selected parameter.
    avnllh : dict
        A dictionary where keys correspond to mass ordering indices (0 for NO, 1 for IO),
        and values are N-dimensional arrays of AvNLL values across the parameter grid.
    param_names : list
        List of parameter names.
    """
    param_to_suffix = {'delta':'dcp', 'dm2':'dm2', 'sin223':'sin223', 'sin213':'sin213'} #Finish for all params and for 2D case
    mo_to_suffix = {0:'', 1:'_IH'}
    avnllh = {}

    if mo=='both':
        for imo in [0, 1]: 
            bin_edges, double_avnllh,  _ = read_histogram(os.path.join(indir, filename), f'SA_{param_to_suffix[param]}{mo_to_suffix[imo]}')
            double_avnllh = double_avnllh.reshape(1, *double_avnllh.shape)
            avnllh[imo] = double_avnllh/2
            grid = [(bin_edges[i][1:]+bin_edges[i][:-1])/2 for i in range(len(bin_edges))]
            param_names = [param]
    elif mo==0 or mo==1:
        bin_edges, double_avnllh,  _ = read_histogram(os.path.join(indir, filename), f'SA_{param_to_suffix[param]}{mo_to_suffix[mo]}')
        double_avnllh = double_avnllh.reshape(1, *double_avnllh.shape)
        avnllh[mo] = double_avnllh/2
        grid = [(bin_edges[i][1:]+bin_edges[i][:-1])/2 for i in range(len(bin_edges))]
        param_names = [param]
    else:
        raise ValueError ("mo can be equal to 'both', 0 or 1.")
    return grid, avnllh, param_names

def load(file_pattern, mo="both"):
    """
    Load N-dimensional likelihood from MargTemplate outputs. It automaticaly performs merging for parallelised jobs.

    This function reads ROOT files containing a `MargTemplate` tree and processes them to build
    an array of AvNLL values and the corresponding grids for any number of oscillation parameters.

    Parameters:
    -----------
    file_pattern : str
        A file path pattern (e.g., using wildcards) matching the ROOT files to be processed.
    mo : str or int, optional
        Specifies the assumed mass ordering (MO):
        - 'both': load both tested NO and IO.
        - 0: load tested Normal Ordering (NO).
        - 1: load tested Inverted Ordering (IO).
        Default is 'both'.

    Returns:
    --------
    tuple:
        grid : list of numpy.ndarray
            A list containing arrays, each representing the grid points for one oscillation parameter.
        AvNLLtot : dict
            A dictionary where keys correspond to mass ordering indices (0 for NO, 1 for IO),
            and values are N-dimensional arrays of AvNLL values across the parameter grid.
        param_name : list of str
            Names of the oscillation parameters used in the grid.
    """

    # Merge all trees into a single pandas DataFrame
    combined_data = []
    if isinstance(file_pattern, str):
        filenames = glob.glob(file_pattern)
    elif isinstance(file_pattern, list):
        filenames = []
        for pattern in file_pattern:
            filenames.extend(glob.glob(pattern))
    
    nfiles = len(filenames)
    for filename in filenames:
        with uproot.open(filename) as file:
            if "MargTemplate" not in file:
                print(f"MargTemplate tree not found in file {filename}")
                continue
            input_tree = file["MargTemplate"]
            data = input_tree.arrays(library="pd")
            combined_data.append(data)
    df_all_combined = pd.concat(combined_data, ignore_index=True)

    print(f"Number of entries in 'MargTemplate': {df_all_combined.shape[0]}.")

    # Prepare to extract branches
    branches = list(df_all_combined.columns)
    param_name = [branch for branch in branches if branch in osc_param_name]
    noscparams = len(param_name)
    if "mh" in branches:
        mh_values = np.array(df_all_combined["mh"])
        nmhtested = 2
    else:
        mh_values = None
        nmhtested = 1
        
    if nmhtested == 1 and mo not in [0, 1]:
        raise ValueError(
            "Error: Marginalization performed only for one mass ordering hypothesis. "
            "Specify `mo` as 0 (NO) or 1 (IO)."
        )
    if_no_contparams = False
    # Error handling
    if noscparams == 0:
        print("No continous parameters are found. Hopefully you do p-values studies")
        if_no_contparams = True
        noscparams = 1 # Dummy solution when there are no osc. params

    # Read parameter grids and AvNLLtot values
    grid_values = [np.array(df_all_combined[param]) for param in param_name]
    if if_no_contparams:
        grid_values = [np.zeros(len(np.array(df_all_combined["AvNLLtot"])))]
        param_name = ['none']        
        
    AvNLLtot_values = np.array(df_all_combined["AvNLLtot"])
    ntoys=1
    if "ToyXp" in df_all_combined.columns:
        ToyId_values = np.array(df_all_combined["ToyXp"])
        ntoys = np.max(ToyId_values)+1
    
    grid = [np.unique(values) for values in grid_values]

    # Determine grid sizes
    ngrid = [len(g) for g in grid]
    print(f"Grid sizes: {ngrid} for parameters {param_name}")

    AvNLLtot = {i: np.zeros((ntoys, *ngrid)) for i in range(nmhtested)} if nmhtested == 2 else {mo: np.zeros((ntoys, *ngrid))}

    grid_indices = np.array([np.searchsorted(grid[i], grid_values[i]) for i in range(noscparams)]).T
    mh_grid_indices = mh_values.astype(int) if nmhtested == 2 else np.full(len(df_all_combined), mo)

    shift = np.min(AvNLLtot_values)
    AvNLL_exp = np.exp(-(AvNLLtot_values - shift))

    toyid_values = df_all_combined["ToyXp"].astype(int).values
    for i in range(nmhtested):
        mask = mh_grid_indices == i
        np.add.at(AvNLLtot[i], (toyid_values[mask], *grid_indices[mask].T), AvNLL_exp[mask])

    factor = nfiles
    AvNLLtot = {key: -np.log(value / factor) + shift for key, value in AvNLLtot.items()}
    
    return grid, AvNLLtot, param_name

def load_1D_array(file_pattern):
    grid, avnllh, param_name = load(file_pattern)

    AvNLL_pergrid_pertoy = np.stack([avnllh[0], avnllh[1]], axis=1)
    grid_x = grid[0]
    param_name_x = param_name[0]

    return grid_x, AvNLL_pergrid_pertoy, param_name_x

def save_avnll_hist(llh, outdir):
    #Save here like in C++ macro. It is done to use Smear.C. Ideally Smear.C should be rewritten to Python
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


def transform_s2213_to_sin213(grid, param_name):
    new_grid = []
    new_param_name = []
    if 'sin2213' in param_name:
        print('Tranformation from sin2213 to sin213 will be performed')
        for grid_1axis, param_name_1axis in zip(grid, param_name):
            if param_name_1axis == 'sin2213':
                new_grid.append(0.5*(1-(np.sqrt(1 - grid_1axis))))
                new_param_name.append('sin213')
            else:
                new_grid.append(grid_1axis)
                new_param_name.append(param_name_1axis)               
        return new_grid, new_param_name    
    else:
        print('Tranformation will not be done as in your param_name there is not sin2213')
        return grid, param_name

def update_kwargs(default_kwargs, kwargs):
    """
    Updates kwargs by adding missing keys from default_kwargs
    while keeping existing user-defined values intact.
    
    Parameters:
    -----------
    default_kwargs : dict
        Dictionary containing the default keyword arguments.
    kwargs : dict
        Dictionary with user-specified overrides.
    
    Returns:
    --------
    None (modifies kwargs in place).
    """
    for plot_type, defaults in default_kwargs.items():
        if plot_type not in kwargs:
            kwargs[plot_type] = {}
    
        for key, default_args in defaults.items():
            if key not in kwargs[plot_type]:
                kwargs[plot_type][key] = default_args.copy()  # Copy default values
            else:
                for arg, value in default_args.items():
                    if arg not in kwargs[plot_type][key]:  # Only update missing arguments
                        kwargs[plot_type][key][arg] = value

class Loglikelihood:
    """
    A class to compute and visualize 1D or 2D log-likelihood and Δχ² for oscillation parameter analyses.
    
    Attributes:
    -----------
    grid : list of numpy.ndarray
        Parameter grids for the likelihood analysis.
    avnllh : dict
        A dictionary containing AvNLLH values for different mass orderings (keys are 0, 1, or 'both').
    param_name : list of str
        Names of the oscillation parameters corresponding to the grid.
    dchi2 : dict
        Dictionary of Δχ² values for each mass ordering.
    mo : str or int
        Indicates the tested mass ordering hypotheses ('both', 0, or 1).
    """
    
    def __init__(self, grid, avnllh_pertoy, param_name, mo_treat='joint', itoy=0,):
        """
        Initialize the object.
    
        Parameters:
        -----------
        grid : list of numpy.ndarray
            The parameter grids for the likelihood analysis.
        avnllh : dict
            A dictionary containing AvNLLH values for different mass orderings (keys are 0, 1, or 'both').
        param_name : list of str
            Names of the oscillation parameters corresponding to the grid.
        mo_treat : str, optional
            Type of likelihood analysis:
            - 'joint': Consider dchi2(param, mo).
            - 'conditional': Consider dchi2(param|mo).
            Default is 'joint'.
        itoy : index toy entry to be used for nominal dchi2: self.dchi2
        """
        self.__grid = grid
        self.__avnllh = {key: value[itoy] for key, value in avnllh_pertoy.items()} 
        self.__avnllh_pertoy = avnllh_pertoy 
        self.__param_name = param_name
        self.__mo_treat = mo_treat
        self.__min = {}
        
        # Validate 'mo_treat' argument
        if mo_treat not in ['joint', 'conditional']:
            raise ValueError("Invalid mo_treat: choose 'joint' or 'conditional'.")
            
        # Perform swapping of oscillation parameters if necessary to standardize the axes
        if param_name == ['dm2', 'sin223'] or param_name == ['delta', 'sin223']:
            self.__swap_grid_and_param()

        valid_mo_keys = {0, 1}
        if not set(self.__avnllh.keys()).issubset(valid_mo_keys):
            raise ValueError(f"Invalid keys in avnllh. Expected keys: {valid_mo_keys}")
            
        self.__mo = 'both' if len(self.__avnllh.keys()) == 2 else list(self.__avnllh.keys())[0]
        self.__dchi2_pertoy = self.__calculate_dchi2_pertoy(mo_treat)  
        self.__dchi2 = {key: value[itoy] for key, value in self.__dchi2_pertoy.items()} 

    def _check_grid_size(self, other):
        return all(len(self.grid[i]) == len(other.grid[i]) for i in range(len(self.grid)))
        
    def __add__(self, other):
        if isinstance(other, Loglikelihood) and  _check_grid_size(self, other):
            return Loglikelihood(self.grid, self.avnllh_pertoy + other.avnllh_pertoy, self.param_name, self.__mo_treat)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __sub__(self, other):
        if isinstance(other, Loglikelihood) and _check_grid_size(self, other):
            result = {key: self.avnllh_pertoy[key] - other.avnllh_pertoy[key] for key in self.avnllh_pertoy.keys()}
            return Loglikelihood(self.grid, result, self.param_name, self.__mo_treat)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __truediv__(self, other):
        if isinstance(other, Loglikelihood) and _check_grid_size(self, other):
            result = {}
            for key in self.avnllh_pertoy.keys():
                result[key] = divide_arrays(self.avnllh_pertoy[key], other.avnllh_pertoy[key])
            return Loglikelihood(self.grid, result, self.param_name, self.__mo_treat)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __mul__(self, other):
        if isinstance(other,(int, float)):
            result = {key: self.avnllh_pertoy[key]*other for key in self.avnllh_pertoy.keys()}
            return Loglikelihood(self.grid, result, self.param_name, self.__mo_treat)
        raise ValueError("Incompatible operand type or size")

    def __rmul__(self, other):
        return self.__mul__(other)
    
    ###########  
    @property
    def grid(self):
        """Get the parameter grid."""
        return self.__grid

    @property
    def avnllh_pertoy(self):
        """Get the AvNLLH values."""
        return self.__avnllh_pertoy
    
    @property
    def avnllh(self):
        """Get the AvNLLH values."""
        return self.__avnllh

    @property
    def param_name(self):
        """Get the parameter names."""
        return self.__param_name
    
    @property
    def dchi2(self):
        """Get the Δχ² values."""
        return self.__dchi2
    
    @property
    def dchi2_pertoy(self):
        """Get the Δχ² values."""
        return self.__dchi2_pertoy

    @property
    def mo(self):
        """Get the tested mass ordering hypotheses."""
        return self.__mo
        
    @property
    def min(self):
        """Get the tested mass ordering hypotheses."""
        return self.__min

    def __swap_grid_and_param(self):
        """
        Swap grid and parameter names to standardize the parameter axes for plotting.
        This is applied when the grid order is unconventional (e.g., (X,Y) = ['dm2', 'sin223'] instead of ['sin223', 'dm2']).
        """
        self.__grid[0], self.__grid[1] = self.__grid[1], self.__grid[0]
        self.__param_name[0], self.__param_name[1] = self.__param_name[1], self.__param_name[0]
    
        for key in self.__avnllh.keys():
            self.__avnllh[key] = self.__avnllh[key].transpose()
            self.__avnllh_pertoy[key] = self.__avnllh_pertoy[key].swapaxes(1, 2)
    
    
    def __calculate_dchi2_pertoy(self, mo_treat):
        """
        Calculate the Δχ² values for the likelihood analysis.

        Parameters:
        -----------
        mo_treat : str
            Type of likelihood analysis ('joint' or 'conditional').

        Returns:
        --------
        dchi2 : dict
            Dictionary of Δχ² values for each mass ordering.
        """

        avnllh_dict = self.__avnllh_pertoy
            
        if mo_treat == 'joint': # Joint minimization: find the global minimum across all mass orderings
            global_minimum = np.minimum.reduce([ self.__find_minimum_value(avnllh, key) for key, avnllh in avnllh_dict.items()])        
            dchi2_pertoy = {key: 2 * (avnllh - global_minimum[:, np.newaxis]) for key, avnllh in avnllh_dict.items()}
        elif mo_treat == 'conditional': # Conditional minimization: find the minimum for each mass ordering
            dchi2_pertoy = {key: 2 * (avnllh - self.__find_minimum_value(avnllh, key)[:, np.newaxis]) for key, avnllh in avnllh_dict.items()}
        
        return dchi2_pertoy

    def __find_minimum_value(self, avnllh, mo):
        def calculate_x0(bounds):
            return [(bound[1] + bound[0])/2 for bound in bounds]
        
        def get_points(ix0):

            if self.param_name == ['delta']:
                grid_x = self.grid[0]
                grid_x_extended_left = np.concatenate([[2*grid_x[0] - grid_x[1]], grid_x[:-1]])
                grid_x_extended_right = np.concatenate([grid_x[1:], [2*grid_x[-1] - grid_x[-2]]])
        
                ixm = ix0 - 1
                ixm = np.where(ixm < 0, len(grid_x)-1 + ixm, ixm) 
                ym = avnllh[np.arange(avnllh.shape[0]), ixm]
                xm = grid_x_extended_left[ix0]
                
                ixp = ix0 + 1
                ixp = np.where(ixp > len(grid_x)-1 , ixp-len(grid_x)-1, ixp) 
                yp = avnllh[np.arange(avnllh.shape[0]), ixp]
                xp = grid_x_extended_right[ix0]
            elif self.param_name == ['sindelta']:
                ixm = ix0 - 1
                ixm = np.where(ixm < 0, ix0 + 1, ixm) 
                ym = avnllh[np.arange(avnllh.shape[0]), ixm]
                xm = self.grid[0][ixm]
                ixp = ix0 + 1
                ixp = np.where(ixm >= len(self.grid[0]), ix0 - 1, ixp) 
                yp = avnllh[np.arange(avnllh.shape[0]), ixp]
                xp = self.grid[0][ixp]
            else:
                xp = self.grid[0][ix0+1]
                yp = avnllh[np.arange(avnllh.shape[0]), ix0+1]
                xm = self.grid[0][ix0-1]
                ym = avnllh[np.arange(avnllh.shape[0]), ix0-1]
            return xm, ym, xp, yp
            
        if self.ndim() == 1:
            ix0 = np.argmin(avnllh, axis=1)
            x0, y0 = self.grid[0][ix0],   avnllh[np.arange(avnllh.shape[0]), ix0]  

            xm, ym, xp, yp = get_points(ix0)     

            ymin = np.zeros_like(y0)  
            xmin = x0
            mask = ~np.isclose(y0, 0.0)
            xmin[mask], ymin[mask] = find_parabold_vertex(xm[mask], x0[mask], xp[mask], ym[mask], y0[mask], yp[mask])

            self.__min[mo] = xmin
            return np.array(ymin)
    
        elif self.ndim() == 2:
            # Find the flattened indices of the minimum values along the last two axes
            min_indices_flat = np.argmin(avnllh.reshape(avnllh.shape[0], -1), axis=1)
            # Convert flattened indices to 2D indices (ix00, ix10)
            ix00, ix10 = np.unravel_index(min_indices_flat, avnllh.shape[1:])
            # Get the corresponding values from the grid
            x00, x10 = self.grid[0][ix00], self.grid[1][ix10] 
            # Store the result in the __min dictionary for this mo
            self.__min[mo] = (x00, x10)
            # Return the values at the computed indices for each row i
            return avnllh[np.arange(avnllh.shape[0]), ix00, ix10] 
        else:
            ValueError ("3D likelihood?")

    def get_dchi2_min(self, mo):
        return self.__find_minimum(self.dchi2[mo])
    
    def ndim(self):
        """
        Get the dimensionality of the parameter grid.

        Returns:
        --------
        int
            The number of dimensions in the grid.
        """
        return len(self.__grid)

    def plot(self, ax, wtag=False, mo=None, from_pertoy=False, itoy=0, band=False, show_legend=True, show_map=False, cls=['1sigma', '2sigma', '3sigma'], show_contours=True, show_const_critical=True, x_critical_values=None, critical_values=None, plot_surface=False, **kwargs):
        """
        Plot the Δχ² values in 1D or 2D.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to plot on.
        wtag : bool, optional
            If True, include a tag in the plot. Default is False.
        mo : int or str, optional
            Mass ordering to plot. Default is None (all mass orderings).
        show_legend : bool, optional
            If True, show the legend. Default is True.
        show_map : bool, optional
            If True, show the color map in 2D plots. Default is False.
        cls : list of str, optional
            Confidence levels to plot. Default is ['1sigma', '90%', '3sigma'].
        kwargs : dict
            Additional arguments for plotting.
        """
        if self.ndim() == 1 and not from_pertoy:
            first_legend, second_legend = self._plot_1d(ax, wtag, mo, show_legend, show_const_critical, x_critical_values, critical_values, **kwargs)
        elif self.ndim() == 1 and from_pertoy:
            first_legend, second_legend = self._plot_1d_pertoy(ax, wtag, mo, itoy, band, show_legend, show_const_critical, **kwargs)
        elif self.ndim() == 2:
            first_legend, second_legend = self._plot_2d(ax, wtag, mo, show_map, show_contours, cls, show_legend, plot_surface, **kwargs)
        else:
            raise ValueError(f"Plotting is only supported for 1D and 2D delta chi2. But the llh dimension is {self.ndim()}")  
        return first_legend, second_legend
    
    def find_CI(self, nsigma, mo):
        """
        Find confidence intervals at a given significance level.
    
        Parameters:
        -----------
        nsigma : float
            The number of standard deviations for the confidence level.
        mo : int or str
            The mass ordering to consider ('0', '1', or 'both').
    
        Raises:
        -------
        ValueError
            If the dimensionality of the data is not 1D or 2D.
        """
        if self.ndim() == 1:
            return self._find_CI_1d(nsigma, mo)
        elif self.ndim() == 2:
            return self._find_CI_2d(nsigma, mo)
        else:
            raise ValueError(f"find_CI is only supported for 1D and 2D delta chi2. But the llh dimension is {self.ndim()}")     

    def _find_CI_1d(self, crit_calue, mo):
        edges_left = []   
        edges_right = []
        c = crit_calue
    
        # Treat the case if margin points are inside C.I.
        if self.dchi2[mo][0] <= c:
            edges_left.append(self.grid[mo][0])
        if self.dchi2[mo][-1] <= c:
            edges_right.append(self.grid[mo][-1])

        # Find all the margins of C.I.
        for i in range(len(self.grid[0]) - 1):
            y0, y1 = self.dchi2[mo][i], self.dchi2[mo][i + 1]
            if (y0 - c) >= 0 and (y1 - c) <= 0:
                x0, x1 = self.grid[0][i], self.grid[0][i + 1]
                edge_left = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_left.append(edge_left) 
            if (y0 - c) <= 0 and (y1 - c) >= 0:
                x0, x1 = self.grid[0][i], self.grid[0][i + 1]
                edge_right = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_right.append(edge_right)  
        return np.sort(np.array(edges_left+edges_right))

    def _find_CI_2d(self, nsigma, mo):
        pass

    def _plot_1d(self, ax, wtag, mo, show_legend, show_const_critical, x_critical_values, critical_values, **kwargs):

        def plot_CI_shading():
            dense_grid = np.linspace(self.__grid[0][0], self.__grid[0][-1], 50000)
            if mo in [0, 1]:
                for level in critical_values[mo].keys():
                    dchi2_dense = np.interp(dense_grid, self.__grid[0], self.dchi2[mo], kind='cubic')
                    crit_dense = np.interp(dense_grid, x_critical_values, critical_values[mo][level], kind='cubic')
                    mask = dchi2_dense <= crit_dense
                    plt.fill_between(dense_grid, dchi2_dense, 0, where=mask, interpolate=True, color=level_to_color[mo][level], alpha=0.5)
            else:
                for i, key in enumerate(reversed(self.__dchi2.keys())):
                    levels_desc = sorted(critical_values[key].keys(), reverse=True)
                    levels = sorted(critical_values[key].keys(), reverse=False)
                    for level in levels_desc:
                        dchi2_dense = np.interp(dense_grid, self.__grid[0], self.dchi2[key])
                        crit_dense = np.interp(dense_grid, x_critical_values, critical_values[key][level])
                        mask = dchi2_dense <= crit_dense
                        indices = np.vstack([np.where((mask[1:] & ~mask[:-1]))[0] + 1, np.where((~mask[1:] & mask[:-1]))[0] + 1])
                        
                        for index in indices:
                            ax.vlines(dense_grid[index], ymin=0, ymax=dchi2_dense[index], color=level_to_color[key][level], linewidth=1.5)

                        ax.fill_between(dense_grid, dchi2_dense, 0, where=mask, interpolate=True, facecolor=level_to_color[key][level],
                                        alpha=1.,hatch=level_to_hatch[level], edgecolor='white', linewidth=0, zorder=0)

        def set_axes_legend_config(ax):
            global leg_loc_1, leg_loc_2, bbox_to_anchor
            leg_loc_1 = None
            leg_loc_2 = None
            if self.__param_name[0] == 'dm2':
                ax.ticklabel_format(style='scientific', axis='x', scilimits=(-3, 3))
                ax.set_xlim(0.00228, 0.00272)
                ax.set_ylim(0, 25)
                leg_loc_1 = 'upper center'
                ax.set_xticks(np.arange(2.3e-3, 2.8e-3, 0.1e-3))
            elif self.__param_name[0] == 'delta':
                ax.set_xlim(-np.pi, np.pi)
                ax.set_ylim(0, 30)
                ax.set_xticks(np.arange(-3, 4, 1))
                leg_loc_1 = 'upper left'
                leg_loc_2 = 'center left'
                bbox_to_anchor = (0.01, 0.6)
            elif self.__param_name[0] == 'sin223':
                ax.set_xlim(0.38, 0.64)
                ax.set_ylim(0, 25)
                leg_loc_1 = 'upper center'
                leg_loc_2 = 'center'
                bbox_to_anchor = (0.5, 0.6)
            elif self.__param_name[0] == 'sin213':
                #ax.set_xticks(np.arange(0.018, 0.026, 1e-3))
                ax.set_xlim(0.0182, 0.0259)
                ax.set_ylim(0, 30) 
        
        default_kwargs = { 'ax.plot' : {0: {"color": color_mo[0], "label": mo_to_label[0]},
                                    1: {"color": color_mo[1], "label": mo_to_label[1]}}
                     }
        
        update_kwargs(default_kwargs, kwargs)

        if critical_values is not None:
            plot_CI_shading()
                        
        if mo in [0, 1]:
            ax.plot(self.__grid[0], self.__dchi2[mo], **kwargs['ax.plot'][mo])
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][mo])
        else:
            for key in self.__dchi2.keys():
                ax.plot(self.__grid[0], self.__dchi2[key], **kwargs['ax.plot'][key])
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][self.__mo])    

        ax.set_ylim(0)
        ax.set_ylabel(r'$\Delta \chi^2$')
        set_axes_legend_config(ax)
        show_minor_ticks(ax)
        first_legend = None
        second_legend = None
        
        if show_legend:
            first_legend = ax.legend(edgecolor='white', loc=leg_loc_1, frameon=True)
            ax.add_artist(first_legend)
            if critical_values is not None:
                legend_patches = [ mpatches.Patch(facecolor='white', alpha=1., hatch=level_to_hatch[level], 
                                   edgecolor='black', linewidth=1, label=level_to_label[level])
                                   for level in sorted(critical_values[0].keys(), reverse=False)]
                
                fig = ax.figure
                renderer = fig.canvas.get_renderer()
                first_legend_bbox = first_legend.get_window_extent(renderer)
                bbox_transformed = first_legend_bbox.transformed(ax.transAxes.inverted())
                x0, y0, width, height = bbox_transformed.bounds          
                second_legend = ax.legend(handles=legend_patches, 
                                          frameon=True,  ncol=1, fontsize=18, loc=leg_loc_2, bbox_to_anchor=bbox_to_anchor)
                ax.add_artist(second_legend)
                
        if show_const_critical:
            ax.axhline(1, ls='--', color='grey', linewidth=1, zorder=0)
            ax.axhline(4, ls='--', color='grey', linewidth=1, zorder=0)
            ax.axhline(9, ls='--', color='grey', linewidth=1, zorder=0) 

            ax.text(0.03, 1.0, r'1$\sigma$', color='grey', fontsize=10, verticalalignment='bottom', horizontalalignment='left', transform=ax.get_yaxis_transform())
            ax.text(0.03, 4.0, r'2$\sigma$', color='grey', fontsize=10, verticalalignment='bottom', horizontalalignment='left', transform=ax.get_yaxis_transform())
            ax.text(0.03, 9.1, r'3$\sigma$', color='grey', fontsize=10, verticalalignment='bottom', horizontalalignment='left', transform=ax.get_yaxis_transform())
        if wtag:
            ax.set_title(cfg.CONFIG.tag, loc='right')
        return first_legend, second_legend

    def _plot_1d_pertoy(self, ax, wtag, mo, itoy, band, show_legend, show_const_critical, **kwargs):

        dict_to_plot = self.__dchi2_pertoy # should be added as option
        def plot_band(mo):
            
            y_median = np.median(dict_to_plot[mo], axis=0)
            y_low_1sigma, y_high_1sigma = np.percentile(dict_to_plot[mo], [16, 84], axis=0)
            y_low_2sigma, y_high_2sigma = np.percentile(dict_to_plot[mo], [2.5, 97.5], axis=0)
            ax.fill_between(self.grid[0], y_low_2sigma, y_high_2sigma, color="yellow", alpha=0.5, label="2σ")
            ax.fill_between(self.grid[0], y_low_1sigma, y_high_1sigma, color="green", alpha=0.7, label="1σ")
            ax.plot(self.grid[0], y_median, color="black", ls='--', label="Median")
        
        if mo in [0, 1]:
            if band:
                plot_band(mo)
            else:
                ax.plot(self.__grid[0], dict_to_plot[mo][itoy], **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][mo])
        else:
            if band:
                for key in dict_to_plot.keys():
                    plot_band(key)
            else:
                for key in dict_to_plot.keys():
                    ax.plot(self.__grid[0], dict_to_plot[key][itoy], color=color_mo[key], label=mo_to_label[key], **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][self.__mo]) 
        
        if wtag:
            ax.set_title(cfg.CONFIG.tag, loc='right')
        ax.set_ylabel(r'$\Delta \chi^2$')
        show_minor_ticks(ax)
        ax.set_ylim(0)

        if self.__param_name[0] == 'dm2':
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(-3, 3))
            ax.set_xlim(0.00228, 0.00272)
            ax.set_ylim(0, 25)
            leg_loc_1 = 'upper center'
            ax.set_xticks(np.arange(2.3e-3, 2.8e-3, 0.1e-3))
        elif self.__param_name[0] == 'delta':
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(0, 30)
            ax.set_xticks(np.arange(-3, 4, 1))
            leg_loc_1 = 'upper left'
            leg_loc_2 = 'center left'
            bbox_to_anchor = (0.01, 0.6)
        elif self.__param_name[0] == 'sin223':
            ax.set_xlim(0.38, 0.64)
            ax.set_ylim(0, 25)
            leg_loc_1 = 'upper center'
            leg_loc_2 = 'center'
            bbox_to_anchor = (0.5, 0.6)
        elif self.__param_name[0] == 'sin213':
            #ax.set_xticks(np.arange(0.018, 0.026, 1e-3))
            ax.set_xlim(0.0182, 0.0259)
            ax.set_ylim(0, 30) 

        if show_const_critical:
            ax.axhline(1, ls='--', color='grey', linewidth=1, zorder=0)
            ax.axhline(4, ls='--', color='grey', linewidth=1, zorder=0)
            ax.axhline(9, ls='--', color='grey', linewidth=1, zorder=0) 
            x_min, x_max = ax.get_xlim()
            x_pos = x_min + 0.03 * (x_max - x_min) 
            ax.text(x_pos, 1, r'1$\sigma$', color='grey', fontsize=10, verticalalignment='bottom')
            ax.text(x_pos, 4, r'2$\sigma$', color='grey', fontsize=10, verticalalignment='bottom')
            ax.text(x_pos, 9, r'3$\sigma$', color='grey', fontsize=10, verticalalignment='bottom')
            
        return None, None

    def _plot_2d(self, ax, wtag, mo, show_map, show_contours, cls, show_legend, plot_surface, **kwargs):

        default_kwargs = { 'ax.plot'  : {0: {"color": color_mo[0], "label": mo_to_label[0]},
                                    1: {"color": color_mo[1], "label": mo_to_label[1]}},
                          'ax.contour': {0: {'colors': [color_mo[0]]},
                                     1: {'colors': [color_mo[1]]}},
                          'ax.scatter': {0: {'color': [color_mo[0]], 'marker': 'x'},
                                     1: {'color': [color_mo[1]], 'marker': 'x'}},
                          'ax.pcolormesh': {0: {'cmap': rev_afmhot},
                                     1: {'cmap': rev_afmhot}},
                     }
        update_kwargs(default_kwargs, kwargs)

        def get_chi2_critical_values():
            critical_values = []
            coverages = []
            for cl in cls:
                if 'sigma' in cl:
                    z_score = float(cl.replace('sigma', '').strip())
                    coverage = sigma_to_CL(z_score)
                elif '%' in cl:
                    coverage = float(cl.replace('%', '').strip())/100
                else:
                    raise ValueError("cls should be a list, where each element has to have one of the two forms: '<z_score>sigma' or '<CL>%'")  
                coverages.append(round(coverage, 4))
                
                critical_value = round(CL_to_chi2critval(coverage, dof=self.ndim()), 4)
                critical_values.append(critical_value)
            return critical_values, coverages

        def plot_contour_for_mo(mo, **kwargs):
            contour = ax.contour(self.__grid[0], self.__grid[1], self.__dchi2[mo].transpose(), 
                                 levels=critical_values, zorder=1, linestyles=['--', '-', 'dotted'], **kwargs['ax.contour'][mo])
            
            ax.contourf(self.__grid[0], self.__grid[1], self.__dchi2[mo].T, levels=[0, critical_values[1]],  # Fill only the first region
                        zorder=0, alpha=0.25, **kwargs['ax.contour'][mo])

        def plot_surface_for_mo(mo, **kwargs):
            X, Y = np.meshgrid(self.__grid[0], self.__grid[1])
            Z = self.__dchi2[mo].transpose()
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
        def plot_map_for_mo(mo, **kwargs):
            mesh = ax.pcolormesh(self.__grid[0], self.__grid[1], self.__dchi2[mo].transpose(), zorder=0, **kwargs['ax.pcolormesh'][mo])
            cbar = plt.colorbar(mesh, ax=ax)

        def set_lim_ticks(axis, lim_min, lim_max, tick_min=None, tick_max=None, tick_step=None):
            if axis == 'x':
                ax.set_xlim(lim_min, lim_max)
            if axis == 'y':
                ax.set_ylim(lim_min, lim_max) 
            if tick_min is not None and tick_max is not None and tick_step is not None:
                if axis == 'x':
                    ax.set_xlim(lim_min, lim_max)
                    ax.set_xticks(np.arange(tick_min, tick_max, tick_step ))
                if axis == 'y':
                    ax.set_ylim(lim_min, lim_max) 
                    ax.set_yticks(np.arange(tick_min, tick_max, tick_step))
                
        critical_values, coverages = get_chi2_critical_values() 
            
        if show_map and mo is not None:
            plot_map_for_mo(mo, **kwargs)
            ax.scatter(*self.__min[mo], **kwargs['ax.scatter'][mo])
            color = 'white'
        elif show_map and mo is None:
            raise ValueError(f"Tested mo is None. If you want to plot the heat map, mo should be specified")

        if show_contours and not plot_surface:
            if mo in [0, 1]:
                if mo in self.__dchi2.keys():
                    plot_contour_for_mo(mo, **kwargs)
                    ax.scatter(*self.__min[mo], **kwargs['ax.scatter'][mo])
                else:
                    raise ValueError(f"There is not dchi2 with mo={mo}")  
            else:
                for key in self.__dchi2.keys():
                    plot_contour_for_mo(key, **kwargs)
                    ax.scatter(*self.__min[key], **kwargs['ax.scatter'][key])
                    ax.plot([], [], **kwargs['ax.plot'][key])
        
        if plot_surface and mo is not None:
            plot_surface_for_mo(mo, **kwargs)
        elif plot_surface and mo is None:
            raise ValueError(f"Tested mo is None. If you want to plot the surface, mo should be specified")


        first_legend = None
        second_legend = None
        if show_legend and not plot_surface:
            first_legend = ax.legend(edgecolor='white', loc='upper left', frameon=True)
            ax.add_artist(first_legend)
            handles = []
            for coverage in coverages:
                handles.append(ax.plot([], [], label=level_to_label[coverage], ls=level_to_ls[coverage], color='black')[0])
            handles.append(ax.scatter([], [], label='Best-fit',  color='black', marker='x'))  
            second_legend = ax.legend(handles=handles, frameon=True,  ncol=1, fontsize=16, loc='upper right')
            ax.add_artist(second_legend)
        
        
        if wtag:
            ax.set_title(cfg.CONFIG.tag, loc='right')

        ind_to_axis = {0:'x', 1:'y'}

                
        for i in range(2):   
            if self.__param_name[i] == 'dm2':
                ax.ticklabel_format(style='scientific', axis=ind_to_axis[i], scilimits=(-3, 3))
                set_lim_ticks(ind_to_axis[i], 0.00228, 0.00290, 2.3e-3, 2.95e-3, 0.1e-3)
                
            elif self.__param_name[i] == 'delta':
                set_lim_ticks(ind_to_axis[i], -np.pi, np.pi, -3, 4, 1)

            elif self.__param_name[i] == 'sin213':
                ax.ticklabel_format(style='scientific', axis=ind_to_axis[i], scilimits=(-3, 3))
                set_lim_ticks(ind_to_axis[i], 17.5e-3, 27.5e-3)
            
        ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][self.__mo], fontsize=23)
        ax.set_ylabel(osc_param_name_to_xlabel[self.__param_name[1]][self.__mo], fontsize=23)
        show_minor_ticks(ax)

        return first_legend, second_legend
    
        






