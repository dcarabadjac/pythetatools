from array import array
import uproot
import numpy as np
from .global_names import *
from .base_visualisation import *
from .base_analysis import *
import sys
import glob
import pandas as pd


def load(file_pattern, nthrows_per_file, target_nthrows, mo="both"):
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
    for filename in filenames:
        with uproot.open(filename) as file:
            if "MargTemplate" not in file:
                print(f"MargTemplate tree not found in file {filename}")
                continue
            input_tree = file["MargTemplate"]
            data = input_tree.arrays(library="pd")
            combined_data.append(data)
    combined_trees = pd.concat(combined_data, ignore_index=True)

    print(f"Number of entries in 'MargTemplate': {combined_trees.shape[0]}.")

    # Prepare to extract branches
    branches = list(combined_trees.columns)
    param_name = [branch for branch in branches if branch in osc_param_name]
    noscparams = len(param_name)
    if "mh" in branches:
        mh_values = np.array(combined_trees["mh"])
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
    grid_values = [np.round(np.array(combined_trees[param]), 8) for param in param_name]
    AvNLLtot_values = np.array(combined_trees["AvNLLtot"])
    grid = [np.unique(values) for values in grid_values]

    # Determine grid sizes
    ngrid = [len(g) for g in grid]
    print(f"Grid sizes: {ngrid} for parameters {param_name}")

    assert np.prod(ngrid) * nmhtested == combined_trees.shape[0], "Grid size mismatch."

    # Initialize storage for AvNLLtot
    AvNLLtot = (
        {0: np.zeros(ngrid), 1: np.zeros(ngrid)} if nmhtested == 2 else {mo: np.zeros(ngrid)}
    )

    # Fill AvNLLtot arrays
    for entry in range(combined_trees.shape[0]):
        grid_indices = [np.where(grid[i] == grid_values[i][entry])[0][0] for i in range(noscparams)]
        mh_grid_index = int(mh_values[entry]) if nmhtested == 2 else mo
        AvNLLtot[mh_grid_index][tuple(grid_indices)] += np.exp(-AvNLLtot_values[entry])

    # Normalize AvNLLtot
    AvNLLtot = {
        key: -np.log(value) * nthrows_per_file / target_nthrows for key, value in AvNLLtot.items()
    }

    return grid, AvNLLtot, param_name


class loglikelihood:
    def __init__(self, grid, avnllh, param_name, kind='joint'):
        """
        Initialize the object with grid, AvNLLH values, and parameter names.
    
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
        self.__avnllh = avnllh
        self.__param_name = param_name
        
        # Validate 'kind' argument
        if kind not in ['joint', 'conditional']:
            raise ValueError("Invalid kind: choose 'joint' or 'conditional'.")
            
        # Perform swapping of osc params if necessary to standardize the osc. params axes for plotting
        if param_name == ['dm2', 'sin223']:
            self.__swap_grid_and_param()
            
        self.__mo = 'both' if len(self.__avnllh.keys()) == 2 else list(self.__avnllh.keys())[0]
        self.__dchi2 = self.__calculate_dchi2(kind)

    def __swap_grid_and_param(self):
        """
        Swap grid and parameter names to standardize the osc. params axes for plotting.
        """
        # Swap elements in grid and parameter names
        self.__grid[0], self.__grid[1] = self.__grid[1], self.__grid[0]
        self.__param_name[0], self.__param_name[1] = self.__param_name[1], self.__param_name[0]
    
        # Transpose the AvNLLH arrays
        for key in self.__avnllh.keys():
            self.__avnllh[key] = self.__avnllh[key].transpose()
    
    def __calculate_dchi2(self, kind):
        """
        Calculate the Δχ² values
    
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
            global_minimum = min(np.min(array) for array in self.__avnllh.values())
            dchi2 = {key: 2 * (value - global_minimum) for key, value in self.__avnllh.items()}
        elif kind == 'conditional':
            # Conditional minimization: find the minimum for each mass ordering
            dchi2 = {key: 2 * (value - np.min(value)) for key, value in self.__avnllh.items()}
    
        return dchi2

    def ndim(self):
        return len(self.__grid)

    def plot(self, ax, wtag=False, mo=None, show_legend=True, show_map=False, cls=['1sigma', '90%', '3sigma'], **kwargs):
        if self.ndim() == 1:
            self._plot_1d(ax, wtag, mo, show_legend, **kwargs)
        elif self.ndim() == 2:
            self._plot_2d(ax, wtag, mo, show_map, cls, **kwargs)
        else:
            raise ValueError(f"Plotting is only supported for 1D and 2D delta chi2. But the llh dimension is {self.ndim()}")     

    def find_CI(self, nsigma, mo):
        if self.ndim() == 1:
            self._find_CI_1d(nsigma, mo)
        elif self.ndim() == 2:
            self._find_CI_2d(nsigma, mo)
        else:
            raise ValueError(f"find_CI is only supported for 1D and 2D delta chi2. But the llh dimension is {self.ndim()}")     

    def _find_CI_1d(self, nsigma, mo):
        edges_left = []   
        edges_right = []
        c = np.sqrt(nsigma)

        #Treat the case if margin points in inside C.I.
        if self.dchi2[mo][0] <= c:
            edges_left.append(self.grid[0])
        if self.dchi2[mo][-1] <= c:
            edges_right.append(self.grid[-1])

        #Find all the margins of C.I.
        for i in range(len(self.grid) - 1):
            y0, y1 = self.dchi2[mo][i], self.dchi2[mo][i + 1]
            if (y0 - c)>=0 and (y1 - c)<=0:
                x0, x1 = self.grid[i], self.grid[i + 1]
                edge_left = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_left.append(edge_left) 
            if (y0 - c)<=0 and (y1 - c)>=0:
                x0, x1 = self.grid[i], self.grid[i + 1]
                edge_right = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_right.append(edge_right)          
        return edges_left, edges_right

    def _find_CI_2d(self, nsigma, mo):
        edges_left = []   
        edges_right = []
        c = np.sqrt(nsigma)

        #Treat the case if margin points in inside C.I.
        if self.dchi2[mo][0] <= c:
            edges_left.append(self.grid[0])
        if self.dchi2[mo][-1] <= c:
            edges_right.append(self.grid[-1])

        #Find all the margins of C.I.
        for i in range(len(self.grid) - 1):
            y0, y1 = self.dchi2[mo][i], self.dchi2[mo][i + 1]
            if (y0 - c)>=0 and (y1 - c)<=0:
                x0, x1 = self.grid[i], self.grid[i + 1]
                edge_left = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_left.append(edge_left) 
            if (y0 - c)<=0 and (y1 - c)>=0:
                x0, x1 = self.grid[i], self.grid[i + 1]
                edge_right = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_right.append(edge_right)          
        return edges_left, edges_right

    def _plot_1d(self, ax, wtag, mo, show_legend, **kwargs):
        if not mo is None:
            ax.plot(self.__grid[0], self.__dchi2[mo], color=kwargs.pop('color', color_mo[mo]), 
                    label=kwargs.pop('label', mo_to_label[mo]), **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name][mo])
        else:
            for key, value in self.__dchi2.items():
                ax.plot(self.__grid[0], value, color=color_mo[key], label=mo_to_label[key], **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][self.__mo])    
            
        ax.set_ylabel(r'$\Delta \chi^2$')
        show_minor_ticks(ax)
        ax.set_ylim(0)
        if show_legend:
            ax.legend(edgecolor='white')

    def _plot_2d(self, ax, wtag, mo, show_map, cls, **kwargs):
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
            critical_value = round(critical_value_for_cl(coverage, dof=2), 4)
            fmt[critical_value] = f'{coverage*100:.2f} %'
            critical_values.append(critical_value)
            
        if show_map and mo is not None:
            mesh = ax.pcolormesh(self.__grid[0], self.__grid[1], self.__dchi2[mo].transpose(), zorder=0,  **kwargs)
            cbar = plt.colorbar(mesh, ax=ax)    
            color = 'white'
       
        if not mo is None:
            if mo in self.__dchi2.keys():
                contour = ax.contour(self.__grid[0], self.__grid[1], self.__dchi2[mo].transpose(), 
                                     levels=critical_values, colors=color, linewidths=2, 
                                     zorder=1, linestyles=['-', '--', 'dotted'])
            else:
                raise ValueError(f"There is not dchi2 with mo={mo}")  
        else:
            for key, value in self.__dchi2.items():
                contour = ax.contour(self.__grid[0], self.__grid[1], self.__dchi2[key].transpose(), 
                        levels=critical_values, colors=color, linewidths=2, 
                        zorder=1, linestyles=['-', '--', 'dotted'])

        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
        ax.clabel(contour, fontsize=20, fmt=fmt)
        ax.set_xlabel(osc_param_name_to_xlabel[self.__param_name[0]][mo])
        ax.set_ylabel(osc_param_name_to_xlabel[self.__param_name[1]][mo])
        show_minor_ticks(ax)



