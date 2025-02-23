"""
Defines Loglikelihood classes and related functions
"""

from array import array
import uproot
import numpy as np
from .global_names import *
from .base_visualisation import *
from .base_analysis import *
import sys
import glob
import pandas as pd
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import minimize
import matplotlib.patches as mpatches

import ROOT #is needed to write files



def load(file_pattern, mo="both"):
    """
    Load N-dimensional likelihood from MargTemplate output.

    This function reads ROOT files containing a `MargTemplate` tree and processes them to build
    an array of AvNLL values and the corresponding grids for any number of oscillation parameters.

    Parameters:
    -----------
    file_pattern : str
        A file path pattern (e.g., using wildcards) matching the ROOT files to be processed.
    nthrows_per_file : int
        Number of throws used in the PTMargTemplate for this file.
    target_nthrows : int
        Target number of throws. It should be not larger that nthrows_per_file*(number of files)
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
    filenames = glob.glob(file_pattern)
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

    # Error handling
    if noscparams == 0:
        raise ValueError("Error: Continuous oscillation parameters not found in the tree.")
    if nmhtested == 1 and mo not in [0, 1]:
        raise ValueError(
            "Error: Marginalization performed only for one mass ordering hypothesis. "
            "Specify `mo` as 0 (NO) or 1 (IO)."
        )

    # Read parameter grids and AvNLLtot values
    grid_values = [np.array(df_all_combined[param]) for param in param_name]
    AvNLLtot_values = np.array(df_all_combined["AvNLLtot"])
    ntoys=1
    if "ToyXp" in df_all_combined.columns:
        ToyId_values = np.array(df_all_combined["ToyXp"])
        ntoys = np.max(ToyId_values)+1
    
    grid = [np.unique(values) for values in grid_values]

    # Determine grid sizes
    ngrid = [len(g) for g in grid]
    print(f"Grid sizes: {ngrid} for parameters {param_name}")

    assert np.prod(ngrid) * nmhtested * ntoys * nfiles  == df_all_combined.shape[0], f"Grid size mismatch: {np.prod(ngrid)}*{nmhtested}*{ntoys}*{nfiles}=? {df_all_combined.shape[0]}"

    # Initialize storage for AvNLLtot
    """
    if "ToyXp" in df_all_combined.columns: 
        AvNLLtot = {0: np.zeros((ntoys, ngrid)), 1: np.zeros((ntoys, ngrid))} if nmhtested == 2 else {mo: np.zeros((ntoys, ngrid))}
    else:
        AvNLLtot = {0: np.zeros(ngrid), 1: np.zeros(ngrid)} if nmhtested == 2 else {mo: np.zeros(ngrid)}

    # Fill AvNLLtot arrays
    for entry in range(df_all_combined.shape[0]):
        grid_indices = [np.where(grid[i] == grid_values[i][entry])[0][0] for i in range(noscparams)]
        mh_grid_index = int(mh_values[entry]) if nmhtested == 2 else mo
        toyid = int(ToyId_values[entry])
        AvNLLtot[mh_grid_index][toyid][tuple(grid_indices)] += np.exp(-(AvNLLtot_values[entry] - AvNLLtot_values[0]))
    # Normalize AvNLLtot
    AvNLLtot = {
        key: -np.log(value * nthrows_per_file / target_nthrows) + AvNLLtot_values[0] for key, value in AvNLLtot.items()
    }
    """
    if "ToyXp" in df_all_combined.columns: 
        AvNLLtot = {i: np.zeros((ntoys, *ngrid)) for i in range(nmhtested)} if nmhtested == 2 else {mo: np.zeros((ntoys, *ngrid))}
    else:
        AvNLLtot = {i: np.zeros(ngrid) for i in range(nmhtested)} if nmhtested == 2 else {mo: np.zeros(ngrid)}
    
    # Получаем индексы для сетки параметров
    grid_indices = np.array([np.searchsorted(grid[i], grid_values[i]) for i in range(noscparams)]).T
    mh_grid_indices = mh_values.astype(int) if nmhtested == 2 else np.full(len(df_all_combined), mo)
    
    # Вычисляем экспоненту заранее
    AvNLL_exp = np.exp(-(AvNLLtot_values - AvNLLtot_values[0]))
    
    if "ToyXp" in df_all_combined.columns:
        toyid_values = df_all_combined["ToyXp"].astype(int).values
        for i in range(nmhtested):
            mask = mh_grid_indices == i
            np.add.at(AvNLLtot[i], (toyid_values[mask], *grid_indices[mask].T), AvNLL_exp[mask])
    else:
        for i in range(nmhtested):
            mask = mh_grid_indices == i
            np.add.at(AvNLLtot[i], tuple(grid_indices[mask].T), AvNLL_exp[mask])
    
    # Нормализация
    factor = nfiles
    AvNLLtot = {key: -np.log(value / factor) + AvNLLtot_values[0] for key, value in AvNLLtot.items()}
    
    return grid, AvNLLtot, param_name

def save_avnll_hist(llh, outdir):

    class OscParam:
        not_defined = -1
        sin2213 = 0
        deltaCP = 1
        dm32 = 2
        sin223 = 3
        sin213 = 4
        OscParamCount = 5

    # Create TObjString for grid parameter name
    OscParamName = ["sin2213", "delta", "dm32", "sin223", "sin213"]

    def get_enum_value(param_name):
        mapping = {
            "not_defined": OscParam.not_defined,
            "sin2213": OscParam.sin2213,
            "delta": OscParam.deltaCP,
            "dm2": OscParam.dm32,
            "sin223": OscParam.sin223,
            "sin213": OscParam.sin213,
        }
        return mapping[param_name]
    
    mo_to_suffix = {0: '', 1: '_IH'}
    for mo in llh.dchi2.keys():
        x_bin_centers = llh.grid[0]
        x_contents = 2*llh.avnllh[mo] #we store -2lnL in cont histogram

        # Create bin edges from bin centers
        x_bin_edges = np.zeros(len(x_bin_centers) + 1)
        x_bin_edges[0] = x_bin_centers[0] - (x_bin_centers[1] - x_bin_centers[0]) / 2  # First edge
        x_bin_edges[1:-1] = (x_bin_centers[:-1] + x_bin_centers[1:]) / 2  # Intermediate edges
        x_bin_edges[-1] = x_bin_centers[-1] + (x_bin_centers[-1] - x_bin_centers[-2]) / 2  # Last edge

        # Create a TH1D histogram
        hist = ROOT.TH1D(f"cont", "cont", len(x_bin_edges) - 1, x_bin_edges)

        hist.GetXaxis().SetTitle(osc_param_title[llh.param_name[0]][mo])
        hist.GetYaxis().SetTitle(r'$-2 \ln{L}$')

        # Fill the histogram with contents
        for i in range(len(x_contents)):
            hist.SetBinContent(i + 1, x_contents[i])  # Fill bin contents (1-indexed in ROOT)

        filename = f"{outdir}/hist{mo_to_suffix[mo]}.root"
        # Save the histogram to a ROOT file
        output_file = ROOT.TFile(f"{outdir}/hist{mo_to_suffix[mo]}.root", "RECREATE")
        hist.Write()
 
        gridParam = get_enum_value(llh.param_name[0])
        # Create TParameter<int>
        pGridParam = ROOT.TParameter(int)(f"gridParam", gridParam)
        pGridParam.Write()
            
        gridParamName = ROOT.TObjString(OscParamName[gridParam])
        gridParamName.Write("gridParamName")

        print(f'Objects written in {filename}')
        output_file.Close()

    print("Histograms saved to ROOT files.")


def transform_s2213_to_sin213(grid, param_name):

    new_grid = []
    new_param_name = []
    for grid_1axis, param_name_1axis in zip(grid, param_name):
        if param_name_1axis == 'sin2213':
            new_grid.append(0.5*(1-(np.sqrt(1 - grid_1axis))))
            new_param_name.append('sin213')
            return new_grid, new_param_name
        else:
            print('Tranformation will not be done as in your param_name there is not sin2213')
            return grid, param_name

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
    
    def __init__(self, grid, avnllh, param_name, kind='joint'):
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
        kind : str, optional
            Type of likelihood analysis:
            - 'joint': Consider dchi2(param, mo).
            - 'conditional': Consider dchi2(param|mo).
            Default is 'joint'.
        """
        self.__grid = grid
        self.__avnllh = {key: value[0] for key, value in avnllh.items()} 
        self.__avnllh_pertoy = avnllh 
        self.__param_name = param_name
        self.__kind = kind
        
        # Validate 'kind' argument
        if kind not in ['joint', 'conditional']:
            raise ValueError("Invalid kind: choose 'joint' or 'conditional'.")
            
        # Perform swapping of oscillation parameters if necessary to standardize the axes
        if param_name == ['dm2', 'sin223']:
            self.__swap_grid_and_param()

        valid_mo_keys = {0, 1, 'both'}
        if not set(self.__avnllh.keys()).issubset(valid_mo_keys):
            raise ValueError(f"Invalid keys in avnllh. Expected keys: {valid_mo_keys}")
            
        self.__mo = 'both' if len(self.__avnllh.keys()) == 2 else list(self.__avnllh.keys())[0]
        #self.__dchi2_pertoy = {key: value[0] for key, value in self.__dchi2_pertoy.items()} Not finished!
        self.__dchi2 = self.__calculate_dchi2(kind) 
        
    ### Nice special methods definition
    def __add__(self, other):
        if isinstance(other, Loglikelihood) and all(len(self.grid[i]) == len(other.grid[i]) for i in range(len(self.grid))) and self.__param_name == other.__param_name:
            return Loglikelihood(self.grid, self.avnllh + other.avnllh, self.param_name, self.__kind)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __sub__(self, other):
        if isinstance(other, Loglikelihood) and \
        all(len(self.grid[i]) == len(other.grid[i]) for i in range(len(self.grid))) and \
        self.__param_name == other.__param_name:
            
            result = {key: self.avnllh[key] - other.avnllh[key] for key in self.avnllh.keys()}
            return Loglikelihood(self.grid, result, self.param_name, self.__kind)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __truediv__(self, other):
        if isinstance(other, Loglikelihood) and \
        all(len(self.grid[i]) == len(other.grid[i]) for i in range(len(self.grid))) and \
        self.__param_name == other.param_name:
            
            result = {}
            for key in self.avnllh.keys():
                where_zeros = other.avnllh[key]==0
                ratio = np.zeros_like(self.avnllh[key])
                ratio[where_zeros] = 0
                ratio[~where_zeros] = self.avnllh[key][~where_zeros] / self.avnllh[key][~where_zeros]
                result[key] = ratio
            return Loglikelihood(self.grid, result, self.param_name, self.__kind)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __mul__(self, number):
        result = {key: self.avnllh[key]*number for key in self.avnllh.keys()}
        return Loglikelihood(self.grid, result, self.param_name, self.__kind)

    
    ###########
    def __find_minimum(self, array):
        def calculate_x0(bounds):
            return [(bound[1] + bound[0])/2 for bound in bounds]
            
        if self.ndim() == 1:
            interpolator = interp1d(self.__grid[0], array, kind='linear')
            bounds = [(self.__grid[0][0], self.__grid[0][-1])]
            x0 = self.grid[0][np.argmin(array)]
            result = minimize(interpolator, x0=x0, bounds=bounds)
            self.__min = result.x
            return result.fun
    
        elif self.ndim() == 2:
            interpolator = RectBivariateSpline(self.__grid[0], self.__grid[1], array)
            # Define the function to minimize
            def func_to_minimize(xy):
                xi, yi = xy
                return interpolator(xi, yi)[0, 0] 
            bounds = [(self.__grid[0][0], self.__grid[0][-1]), (self.__grid[1][0], self.__grid[1][-1])]
            x0 = calculate_x0(bounds)
            result = minimize(func_to_minimize, x0=x0, bounds=bounds)
            self.__min = result.x
            return result.fun
    
    @property
    def grid(self):
        """Get the parameter grid."""
        return self.__grid
    
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
    def mo(self):
        """Get the tested mass ordering hypotheses."""
        return self.__mo

    def __swap_grid_and_param(self):
        """
        Swap grid and parameter names to standardize the parameter axes for plotting.
        This is applied when the grid order is unconventional (e.g., (X,Y) = ['dm2', 'sin223'] instead of ['dm2', 'sin223']).
        """
        self.__grid[0], self.__grid[1] = self.__grid[1], self.__grid[0]
        self.__param_name[0], self.__param_name[1] = self.__param_name[1], self.__param_name[0]
    
        for key in self.__avnllh.keys():
            self.__avnllh[key] = self.__avnllh[key].transpose()
    
    def __calculate_dchi2(self, kind):
        """
        Calculate the Δχ² values for the likelihood analysis.

        Parameters:
        -----------
        kind : str
            Type of likelihood analysis ('joint' or 'conditional').

        Returns:
        --------
        dchi2 : dict
            Dictionary of Δχ² values for each mass ordering.
        """
        if kind == 'joint':
            # Joint minimization: find the global minimum across all mass orderings
            global_minimum = min(self.__find_minimum(array) for array in self.__avnllh.values())
            dchi2 = {key: 2 * (avnllh - global_minimum) for key, avnllh in self.__avnllh.items()}
        elif kind == 'conditional':
            # Conditional minimization: find the minimum for each mass ordering
            dchi2 = {key: 2 * (avnllh - self.__find_minimum(avnllh)) for key, avnllh in self.__avnllh.items()}
        
        return dchi2

    def ndim(self):
        """
        Get the dimensionality of the parameter grid.

        Returns:
        --------
        int
            The number of dimensions in the grid.
        """
        return len(self.__grid)

    def plot(self, ax, wtag=False, mo=None, show_legend=True, show_map=False, cls=['1sigma', '90%', '3sigma'], show_const_critical=True, x_critical_values=None, critical_values=None, **kwargs):
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
        if self.ndim() == 1:
            self._plot_1d(ax, wtag, mo, show_legend, show_const_critical, x_critical_values, critical_values, **kwargs)
        elif self.ndim() == 2:
            self._plot_2d(ax, wtag, mo, show_map, cls, **kwargs)
        else:
            raise ValueError(f"Plotting is only supported for 1D and 2D delta chi2. But the llh dimension is {self.ndim()}")     
    
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

    def _find_CI_1d(self, nsigma, mo):
        edges_left = []   
        edges_right = []
        c = nsigma**2
    
        # Treat the case if margin points are inside C.I.
        if self.dchi2[mo][0] <= c:
            edges_left.append(self.grid[0])
        if self.dchi2[mo][-1] <= c:
            edges_right.append(self.grid[-1])
    
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
        return edges_left, edges_right

    def _find_CI_2d(self, nsigma, mo):
        pass

    def _plot_1d(self, ax, wtag, mo, show_legend, show_const_critical, x_critical_values, critical_values, **kwargs):
            
        default_kwargs = {}
        if mo in [0, 1]:
            default_kwargs = {
                'color': color_mo[mo], 
                'label': mo_to_label[mo]   
            }
        default_kwargs.update(kwargs)

        if show_const_critical:
            ax.axhline(1, ls='--', color='grey', linewidth=1)
            ax.axhline(4, ls='--', color='grey', linewidth=1)
            ax.axhline(9, ls='--', color='grey', linewidth=1) 
            ax.text(ax.get_xlim()[0]+0.1*np.abs(ax.get_xlim()[0]), 1, r'1$\sigma$', color='grey', fontsize=10, verticalalignment='bottom')
            ax.text(ax.get_xlim()[0]+0.1*np.abs(ax.get_xlim()[0]), 4, r'2$\sigma$', color='grey', fontsize=10, verticalalignment='bottom')
            ax.text(ax.get_xlim()[0]+0.1*np.abs(ax.get_xlim()[0]), 9, r'3$\sigma$', color='grey', fontsize=10, verticalalignment='bottom')

        if critical_values is not None:
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

                        ax.fill_between(dense_grid, dchi2_dense, 0, where=mask, interpolate=True, facecolor=level_to_color[key][level], alpha=1.,hatch=level_to_hatch[level], edgecolor='white', linewidth=0, zorder=0)
                        
        if mo in [0, 1]:
            ax.plot(self.__grid[0], self.__dchi2[mo], **default_kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][mo])
        else:
            for key in self.__dchi2.keys():
                ax.plot(self.__grid[0], self.__dchi2[key], color=color_mo[key], label=mo_to_label[key], **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][self.__mo])    
                
        if self.__param_name[0] == 'dm2':
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(-3, 3))
        ax.set_ylabel(r'$\Delta \chi^2$')
        show_minor_ticks(ax)
        ax.set_ylim(0)
        if show_legend:
            first_legend = ax.legend(edgecolor='white', loc='best', frameon=True)
            ax.add_artist(first_legend)
        
            if critical_values is not None:
                legend_patches = [
                    mpatches.Patch(facecolor='white', alpha=1., hatch=level_to_hatch[level], 
                                   edgecolor='black', linewidth=1, label=level_to_label[level])
                    for level in levels
                ]
                
                fig = ax.figure
                renderer = fig.canvas.get_renderer()
                first_legend_bbox = first_legend.get_window_extent(renderer)
                bbox_transformed = first_legend_bbox.transformed(ax.transAxes.inverted())
                x0, y0, width, height = bbox_transformed.bounds          
                second_legend = ax.legend(handles=legend_patches, 
                                          frameon=True, bbox_to_anchor=(x0 , y0 - 0.02), ncol=1)
                ax.add_artist(second_legend)


    def _plot_2d(self, ax, wtag, mo, show_map, cls, contour_kwargs=None, scatter_kwargs=None, map_kwargs=None):

        if mo in [0, 1]:
            default_contour_kwargs = {
                'colors': color_mo[mo]
                }
            default_scatter_kwargs = {
                'color': color_mo[mo],
                'marker': 'x'
                }
            default_map_kwargs = {
                'cmap': rev_afmhot
                }
        if contour_kwargs is not None:
            default_contour_kwargs.update(contour_kwargs)
        if scatter_kwargs is not None:
            default_scatter_kwargs.update(scatter_kwargs)
        if map_kwargs is not None:
            default_map_kwargs.update(map_kwargs)

        def get_chi2_critical_values():
            critical_values = []
            fmt = {}
            color = color_mo[mo]
            for cl in cls:
                if 'sigma' in cl:
                    z_score = float(cl.replace('sigma', '').strip())
                    coverage = cl_for_sigma(z_score)
                elif '%' in cl:
                    coverage = float(cl.replace('%', '').strip())/100
                else:
                    raise ValueError("cls should be a list, where each element has to have one of the two forms: '<z_score>sigma' or '<CL>%'")  
                critical_value = round(critical_value_for_cl(coverage, dof=self.ndim()), 4)
                fmt[critical_value] = f'{coverage*100:.2f} %'
                critical_values.append(critical_value)
            return critical_values, fmt

        def plot_contour_for_mo(mo):
            contour = ax.contour(self.__grid[0], self.__grid[1], self.__dchi2[mo].transpose(), 
                                 levels=critical_values, zorder=1, linestyles=['-', '--', 'dotted'], **default_contour_kwargs)
            ax.clabel(contour, fontsize=20, fmt=fmt)
            
        def plot_map_for_mo(mo):
            mesh = ax.pcolormesh(self.__grid[0], self.__grid[1], self.__dchi2[mo].transpose(), zorder=0, **default_map_kwargs)
            cbar = plt.colorbar(mesh, ax=ax)
                
        critical_values, fmt = get_chi2_critical_values() #fmt necessary to set nice labels on the contours
            
        if show_map and mo is not None:
            plot_map_for_mo(self, mo)
            plt.scatter(*self.__min, **default_scatter_kwargs)
            color = 'white'
        elif show_map and mo is None:
            raise ValueError(f"Tested mo is None. If you want to plot the heat map, mo should be specified")
       
        if mo is not None:
            if mo in self.__dchi2.keys():
                plot_contour_for_mo(mo)
                plt.scatter(*self.__min, **default_scatter_kwargs)

            else:
                raise ValueError(f"There is not dchi2 with mo={mo}")  
        else:
            for mo in self.__dchi2.keys():
                plot_contour_for_mo(mo)
                plt.scatter(*self.__min, **default_scatter_kwargs)

        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
        ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][mo])
        ax.set_ylabel(osc_param_name_to_xlabel[self.__param_name[1]][mo])
        show_minor_ticks(ax)






