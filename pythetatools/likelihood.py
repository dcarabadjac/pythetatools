from array import array
import uproot
import numpy as np
from .global_names import *
from .base_visualisation import *
from .base_analysis import *
import sys
import glob
import pandas as pd


def load_old(file_pattern, dim, nthrows_per_file, target_nthrows, mo='both'):
    """Loads the MargTemplates files with given file pattern 

    Parameters
    ----------
    file_pattern : string
        The file patern of the files to be loaded
    dim : Likelihood dimension: 1D or 2D

    Returns
    ------
    grid, AvNLLtot, param_name 
        An object of ToyXp class containt all the asimovs stored in the root file
    """
    if dim==1:
        return load_1D(file_pattern, nthrows_per_file, target_nthrows, mo=mo)
    elif dim==2:
        return load_2D(file_pattern, nthrows_per_file, target_nthrows, mo=mo)
    else:
        raise ValueError("The implementation is only realised for dim=1 or dim=2.")

def load_1D(file_pattern, nthrows_per_file, target_nthrows, mo='both'):
    """
    Load 1D likelihood from MargTemplate output.

    This function reads ROOT files containing a `MargTemplate` tree, processes the files to
    build the array of AvNLL values and the corresdinds grids

    Parameters:
    -----------
    file_pattern : str
        A file path pattern matching the ROOT files to be processed.
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
            A list containing two arrays, each representing the grid points for one of the 
            two oscillation parameters.
        AvNLLtot : dict
            A dictionary where the keys correspond to mass ordering indices (0 for NO, 1 for IO), 
            and the values are 1D arrays of AvNLL values across the parameter grid.
        param_name : list of str
            Names of the oscillation parameters used in the grid.
    """
    #Merge all the trees to one pandas dataframe
    combined_data = [] 
    filenames = glob.glob(file_pattern)
    for filename in filenames:
        with uproot.open(filename) as file:
            if 'MargTemplate' not in file:
                print(f"MargTemplate tree not found in file {filename}")
                continue 
            input_tree = file['MargTemplate']
            data = input_tree.arrays(library="pd")
            combined_data.append(data)
    combined_trees = pd.concat(combined_data, ignore_index=True)

    nEntries = combined_trees.shape[0]
    print(f"Number of entries in 'MargTemplate': {combined_trees.shape[0]}.")

    # Prepare to extract branches
    branches = list(combined_trees.columns)
    noscparams = 0
    ndiscrparams = 0
    param_name = []
    
    # Check for oscillation parameter and 'mh' branch
    for branch_name in branches:
        if branch_name in osc_param_name:
            noscparams += 1
            print(f"Grid for oscillation parameter found: {branch_name}")
            param_name = branch_name
        elif branch_name == 'mh':
            ndiscrparams = 1
            print("Grid for mh found")
    
    # Error handling for multiple or missing oscillation parameters
    if noscparams > 1:
        raise ValueError(f"Error: Number of continuous osc. params = {noscparams} is greater than 1 but expected 1D dchi2.")
    elif noscparams == 0:
        raise ValueError("Error: Continuous osc. parameter not found in the tree.")

    if ndiscrparams==0 and not (mo==1 or mo==0):
        raise ValueError("Error: The marginalisation was performed only for one mass ordering hypothesis. Please specisy in argument mo the assumed MO: mo=0 for NO, mo=1 for IO")
    
    # Read the 'mh' values if present
    if ndiscrparams == 1:
        mh_values = np.array(combined_trees['mh'])

    # Read the parameter grid and AvNLLtot values
    grid_values = np.round(np.array(combined_trees[param_name]), 8)
    AvNLLtot_values = np.array(combined_trees["AvNLLtot"]) 
    grid = np.unique(grid_values) #Returns the sorted unique elements of an array.

    # Determine the number of grid points for the parameter

    ngrid = grid.size
    print(f"Number of grid points for {param_name} = {ngrid}")
        
    assert ngrid*ndiscrparams*target_nthrows//nthrows_per_file != combined_trees.shape[0], "Problem in grids size determination"
    
    # Initialize storage for AvNLLtot
    AvNLLtot = {0: np.zeros(ngrid), 1: np.zeros(ngrid)} if ndiscrparams == 1 else {mo: np.zeros(ngrid)}
    
    # Fill in the arrays based on the entries. This does not depend on the order of the files reading
    for entry in range(nEntries):
        mh_grid_index = int(mh_values[entry]) if ndiscrparams == 1 else mo
        osc_param_index = np.where(grid==grid_values[entry])[0][0]
        AvNLLtot[mh_grid_index][osc_param_index] += np.exp(-AvNLLtot_values[entry]) #to account 

    AvNLLtot = {key: -np.log(value)*nthrows_per_file/target_nthrows for key, value in AvNLLtot.items()}

    return grid, AvNLLtot, param_name


def load_2D(file_pattern, nthrows_per_file, target_nthrows,  mo='both'):
    """
    Load 2D likelihood from MargTemplate output.

    This function reads ROOT files containing a `MargTemplate` tree, processes the files to
    build the array of AvNLL values and the corresdinds grids

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
            A list containing two arrays, each representing the grid points for one of the 
            two oscillation parameters.
        AvNLLtot : dict
            A dictionary where the keys correspond to mass ordering indices (0 for NO, 1 for IO), 
            and the values are 2D arrays of AvNLL values across the parameter grid.
        param_name : list of str
            Names of the oscillation parameters used in the grid.
    """
    #Merge all the trees to one pandas dataframe
    combined_data = [] 
    filenames = glob.glob(file_pattern)
    for filename in filenames:
        with uproot.open(filename) as file:
            if 'MargTemplate' not in file:
                print(f"MargTemplate tree not found in file {filename}")
                continue 
            input_tree = file['MargTemplate']
            data = input_tree.arrays(library="pd")
            combined_data.append(data)
    combined_trees = pd.concat(combined_data, ignore_index=True)

    nEntries = combined_trees.shape[0]
    print(f"Number of entries in 'MargTemplate': {combined_trees.shape[0]}.")

    # Prepare to extract branches
    branches = list(combined_trees.columns)
    noscparams = 0
    ndiscrparams = 0
    param_name = []
    
    # Check for oscillation parameter and 'mh' branch
    for branch_name in branches:
        if branch_name in osc_param_name:
            noscparams += 1
            print(f"Grid for oscillation parameter found: {branch_name}")
            param_name.append(branch_name)
        elif branch_name == 'mh':
            ndiscrparams = 1
            print("Grid for mh found")
    
    # Error handling for multiple or missing oscillation parameters
    if noscparams > 2:
        raise ValueError(f"Error: Number of continuous osc. params = {noscparams} is greater than 2 but expected 2D dchi2.")
    if noscparams == 1:
        raise ValueError(f"Error: Number of continuous osc. params = {noscparams}, use instead dim=1")
    elif noscparams == 0:
        raise ValueError("Error: Continuous osc. parameter not found in the tree.")

    if not (ndiscrparams==0 and (mo==1 or mo==0)):
        raise ValueError("Error: The marginalisation was performed only for one mass ordering hypothesis. Please specisy in argument mo the assumed MO: mo=0 for NO, mo=1 for IO")
    
    # Read the 'mh' values if present
    if ndiscrparams == 1:
        mh_values = np.array(combined_trees['mh'])

    # Read the parameter grid and AvNLLtot values
    grid_values = [None]*2
    grid = [None]*2
    grid_values[0] = np.round(np.array(combined_trees[param_name[0]]), 8)
    grid_values[1] = np.round(np.array(combined_trees[param_name[1]]), 8)  
    AvNLLtot_values = np.array(combined_trees["AvNLLtot"]) 
    grid[0] = np.unique(grid_values[0]) #Returns the sorted unique elements of an array.
    grid[1] = np.unique(grid_values[1]) #Returns the sorted unique elements of an array.

    # Determine the number of grid points for the parameter
    ngrid = [None]*2
    for i in range(2):
        ngrid[i] = grid[i].size
        print(f"Number of grid points for {param_name[i]} = {ngrid[i]}")
        
    assert ngrid[0]*ngrid[1]*ndiscrparams != combined_trees.shape[0], "Problem in grids size determination"
    
    # Initialize storage for AvNLLtot
    AvNLLtot = {0: np.zeros((ngrid[0], ngrid[1])), 1: np.zeros((ngrid[0], ngrid[1]))} if ndiscrparams == 1 else {mo: np.zeros((ngrid[0], ngrid[1]))}
    
    # Fill in the arrays based on the entries. This does not depend on the order of the files reading
    for entry in range(nEntries):
        osc_param_index = [None]*2
        mh_grid_index = int(mh_values[entry]) if ndiscrparams == 1 else mo
        osc_param_index[0] = np.where(grid[0]==grid_values[0][entry])[0][0]
        osc_param_index[1] = np.where(grid[1]==grid_values[1][entry])[0][0]
        AvNLLtot[mh_grid_index][osc_param_index[0]][osc_param_index[1]] += np.exp(-AvNLLtot_values[entry]) #to account 

    AvNLLtot = {key: -np.log(value)*nthrows_per_file/target_nthrows for key, value in AvNLLtot.items()}

    return combined_trees, grid, AvNLLtot, param_name

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
        Number of throws per file used in the calculation.
    target_nthrows : int
        Target number of throws for normalization.
    mo : str or int, optional
        Specifies the assumed mass ordering (MO):
        - 'both': load both tested NO and IO.
        - 0: load tested Normal Ordering (NO).
        - 1: load tested Inverted Ordering (IO).
        Default is 'both'.

    Returns:
    --------
    tuple:
        combined_trees : pandas.DataFrame
            A combined dataframe of all input data.
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




def swap_elements(lst):
    """Helper function to swap the first and second elements of a list."""
    if len(lst) > 1:
        lst[0], lst[1] = lst[1], lst[0]

class loglikelihood:
    def __init__(self, grid, avnllh, param_name, kind='joint'):
        self.__grid = grid
        self.__avnllh = avnllh
        self.__param_name = param_name
        if param_name == ['dm2', 'sin223']:
            swap_elements(self.__grid)
            swap_elements(self.__param_name)
            for key in self.__avnllh.keys():
                self.__avnllh[key] = self.__avnllh[key].transpose()
            
        if kind == 'joint':
            minimum = {key: min(np.min(array) for array in self.__avnllh.values()) for key in self.__avnllh.keys()} 
        elif kind == 'conditional':
            minimum = {key: np.min(self.__avnllh[key]) for key in self.__avnllh.keys()} 
            
        self.__dchi2 = {key: 2*(value-minimum[key]) for key, value in self.__avnllh.items()}
        if len(list(self.__dchi2.keys())) == 2:
            self.__mo = 'both'
        else:
            self.__mo = list(self.__dchi2.keys())[0]

    def ndim(self):
        return len(self.__grid)

    def plot(self, ax, wtag=False, mo=None, show_legend=True, show_map=False, cls=['1sigma', '90%', '3sigma'], **kwargs):
        if self.ndim() == 1:
            self._plot_1d(ax, wtag, mo, show_legend, **kwargs)
        elif self.ndim() == 2:
            self._plot_2d(ax, wtag, mo, show_map, cls, **kwargs)
        else:
            raise ValueError(f"Plotting is only supported for 1D and 2D delta chi2. But your dimension is {self.ndim()}")     


    def find_CI(self, nsigma, mo):
        
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
            ax.plot(self.__grid[0], self.__dchi2[mo], color=kwargs.pop('color', color_mo[mo]), label=kwargs.pop('label', mo_to_label[mo]), **kwargs)
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



