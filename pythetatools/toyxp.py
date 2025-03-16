from .config import *
from .config_samples import *
from .config_visualisation import *
from .base_visualisation import *
from .base_analysis import divide_arrays
from .file_manager import download

import numpy as np
from matplotlib import pyplot as plt
import subprocess
from array import array
import os

from collections import defaultdict, Sequence
import matplotlib.patches as mpatches

import uproot
import ROOT #Ideally should not be used however it is needed for reading/writing ROOT files in some cases


def get_titles(filename):
    """Provides the sample titles from PTGenerateXP output root file
    
    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file
    """
    sample_titles = []
    
    with uproot.open(filename) as file:
        keys = file.keys() # Get list of all keys        
        for key in keys:
            if (file[key].classname.startswith("TH2") or file[key].classname.startswith("TH1")) and not ('_true' in key): #select only rec distr
                sample_titles.append(key[1:-2])
    return sample_titles
    
def get_samples_info(filename, sample_titles=None):
    """Provides information of the samples (the titles and analysis-type's) stored in PTGenerateXp output root file 
    
    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file
    Returns
    ------
    Dictionary 
        A dictionary according to following format: {sample_title: analysis_type}, where sample_title in the title of the sample, analysis_type is the binning of this sample (e-theta, PTheta or Erec)
    """
    samples_dict = {}

    if sample_titles is None:
        sample_titles = get_titles(filename)
        
    with uproot.open(filename) as file:
        for sample_title in sample_titles:
            analysis_type_key = f"{sample_title}_analysis_type" # Get the corresponding analysis type for this sample
            analysis_type = file[analysis_type_key].all_members['fTitle']
            samples_dict[sample_title] = analysis_type
    return samples_dict


def load(filename, kind, itoy=None, breakdown=False, samples_dict=None, sample_titles=None, tobin=True, ntoys=1):
    """Loads a toy/asimov from output root file of PTGenerateXp

    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file
    type : string
        'asimov' - This option allows to load the histograms for each sample from the file
        'toy' - This option allows to load the tree for each sample from the file (the trees will be binned in the histograms)
    itoy : 
        The index of toy to be loaded. It is relevant only for type='toy'
    Returns
    ------
    ToyXp
        The object of ToyXp class which stores either asimov data set and a toy data set

    Examples
    --------
    >>> filename = "/Users/denis.carabadjac/Python/pythetatools/inputs/ToyXp/ToyXP_00000.root"
    >>> toy = pythetatools.toyanalysis.load(filename, type="toy", itoy=1)

    """
    toy = ToyXp() 
    if kind=='asimov':
        toy = load_hists(filename, breakdown, samples_dict, sample_titles)
    elif kind=='toy':
        toy = load_toy(filename, itoy=itoy, tree_title='ToyXp', samples_dict=samples_dict, sample_titles=sample_titles, tobin=tobin, ntoys=ntoys)
    elif kind=='data':
        toy = load_toy(filename, itoy=0, tree_title='Data_Tree', samples_dict=samples_dict, sample_titles=sample_titles, tobin=tobin)
    return toy


def load_hists(filename, breakdown=False, samples_dict=None, sample_titles=None):
    """Loads asimovs from PTGenerateXp output root file 

    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file

    Returns
    ------
    ToyXp 
        An object of ToyXp class containt all the asimovs stored in the root file
    """
    if samples_dict is None:
        samples_dict = get_samples_info(filename, sample_titles)
    toy = ToyXp()
    
    with uproot.open(filename) as file:
        for sample_title, analysis_type in samples_dict.items():
            hist = file[f'h{sample_title}']
            xedges = hist.axis(0).edges()
            z = hist.values()
            if analysis_type =='e-theta' or analysis_type =='PTheta' or analysis_type =='p-theta':
                yedges = hist.axis(1).edges()
                toy.append(Sample([xedges, yedges], z, sample_title, analysis_type))    
            else:
                toy.append(Sample([xedges], z, sample_title, analysis_type))
                
        if breakdown:
            for sample_title, analysis_type in samples_dict.items():
                for inter in interaction_modes:
                    for osc_channel in osc_channels:
                        hist = file[f'h{sample_title}_{inter}_{osc_channel}']
                        xedges = hist.axis(0).edges()
                        z = hist.values()
                        if analysis_type =='e-theta' or analysis_type =='PTheta' or analysis_type =='p-theta':
                            yedges = hist.axis(1).edges()
                            toy.append(Sample([xedges, yedges], z, f'{sample_title}_{inter}_{osc_channel}', analysis_type, sample_title))    
                        else:
                            toy.append(Sample([xedges], z, f'{sample_title}_{inter}_{osc_channel}', analysis_type, sample_title))
        
    return toy

def load_toy(filename, itoy, tree_title, samples_dict=None, sample_titles=None, tobin=True, ntoys=1):
    """Loads and bins a toy from PTGenerateXp output root file 

    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file
    itoy : 
        The index of toy to be loaded.
    Returns
    ------
    ToyXp
        The object of ToyXp class which stores either asimov data set and a toy data set

    """
    toy = ToyXp()
    if samples_dict is None:
        samples_dict = get_samples_info(filename, sample_titles)
                
    if tobin:
        with uproot.open(filename) as file:
            for sample_title, analysis_type in samples_dict.items():
                if analysis_type_to_dim[analysis_type] == '1D':
                    samples = bin_sample1D(*load_tree(filename, sample_title, tree_title, 
                                                     analysis_type, itoy, ntoys), sample_title=sample_title, analysis_type=analysis_type)
                elif analysis_type_to_dim[analysis_type] == '2D':
                    samples = bin_sample2D(*load_tree(filename, sample_title, tree_title,
                                                     analysis_type, itoy, ntoys), sample_title=sample_title, analysis_type=analysis_type)
                for sample in samples:
                    toy.append(sample)
        return toy
    else:
        with uproot.open(filename) as file:
            for sample_title, analysis_type in samples_dict.items():
                if analysis_type_to_dim[analysis_type] == '1D':
                    sample = UnbinnedSample(*load_tree(filename, sample_title, tree_title, 
                                                     analysis_type, itoy), title=sample_title, analysis_type=analysis_type)
                elif analysis_type_to_dim[analysis_type] == '2D':
                    sample = UnbinnedSample(*load_tree(filename, sample_title, tree_title,
                                                     analysis_type, itoy), title=sample_title, analysis_type=analysis_type)
                toy.append(sample)
        return toy

def load_tree(filename, sample_title, tree_title, analysis_type, itoy, ntoys=1):
    """Loads a unbinned data for a given sample

    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file
    sample_title : string
        The title of the sample. For example, 'nue1R', 'numubar1R', 'numucc1pi' etc
    itoy : 
        The index of the toy to be loaded.
    Returns
    ------
    Tuple 
        A tuple (x, y) where x(y) is array of values of the observable on the x(y) axis. For 1D sample y is None

    """
    with uproot.open(filename) as file:
        input_tree = file.get(tree_title)
        
        if input_tree is None:
            raise ValueError (f"'{tree_title}' tree not found in file {filename}")

        if analysis_type_to_dim[analysis_type] == '2D':
            xvar_branch_name = f"{analysis_type_to_xvar[analysis_type]}_{sample_title}"
            t_branch_name = f"t_{sample_title}"
            data = input_tree.arrays([xvar_branch_name, t_branch_name], entry_start=itoy, entry_stop=itoy + ntoys)
            xvar = data[xvar_branch_name]
            theta = data[t_branch_name]
            return (list(xvar), list(theta))

        xvar_branch_name = f"{analysis_type_to_xvar[analysis_type]}_{sample_title}"
        data = input_tree.arrays([xvar_branch_name], entry_start=itoy, entry_stop=itoy + ntoys)
        xvar = data[xvar_branch_name]            
        return (list(xvar), )

def load_multiple_hists_from_tree(filename, sample_titles, start_ientry, nentries, tree_title='ToyXp'):
    """Loads multiple histograms from a tree in ROOT file in a ToyXp object for each sample and entry.
    For example, it is used for bievent plots

    Parameters
    ----------
    filename : string
        The filename of the ROOT file containing the tree with histograms.
    sample_titles : list of string
        A list of sample titles corresponding to histogram branches in the tree (e.g., 'nue1R', 'numubar1R', etc.).
    tree_title : string
        The name of the tree within the ROOT file from which the histograms are extracted.
    start_ientry : int
        The starting index for the entries to loop over in the tree.
    nentries : int
        Number the entries to loop over in the tree.

    Returns
    -------
    ToyXp
        An object containing the loaded histograms for each sample title and tree entry, each represented as a Sample.
    """
    # Initialize the ToyXp class or object if needed (assuming it's defined elsewhere)
    toy = ToyXp()
    
    # Open the ROOT file
    file = ROOT.TFile(filename)
    
    # Access the tree
    tree = file.Get(tree_title)  # Replace with the actual name of your tree
    
    # Initialize lists to store bin contents and edges
    all_bin_contents = []
    all_bin_edges = []

    for sample_title in sample_titles:
        # Loop through the specified entries in the tree
        for entry in range(start_ientry, start_ientry+nentries):
            tree.GetEntry(entry)  # Load the entry
            
            #histogram = tree.GetBranch(f'hist_{sample_title}')  # Access the histogram branch by name
            histogram = getattr(tree, f'hist_{sample_title}')
            if histogram:
                # Get bin contents and edges
                bin_contents = [histogram.GetBinContent(i) for i in range(1, histogram.GetNbinsX() + 1)]
                bin_edges = [histogram.GetBinLowEdge(i) for i in range(1, histogram.GetNbinsX() + 2)]  # +2 to include the upper edge of the last bin
                toy.append(Sample([bin_edges], bin_contents, title=f'{sample_title}_{entry}', sample_title=sample_title))

    return toy
    

def bin_sample1D(xvar, sample_title, analysis_type=None, bin_edges=None):
    """Bins a toy in 1D histogram

    Parameters
    ----------
    title : string
        The title of the sample
    xvar : numpy.array
         The array of values of the observable on the x axis
    analysis_type : 
        Binning type for this sample
    Returns
    ------
    Sample1D
        The object of Sample1D class binning xvar array
    """
    if bin_edges is None and analysis_type is not None:
        bin_edges = analysis_type_xedges[analysis_type] 
    elif bin_edges is None and analysis_type is None:
        raise ValueError("Either 'bin_edges' or 'analysis_type' must be provided.")
     
    samples = []
    i = -1
    if len(xvar) > 1:
        for x in xvar: 
            i += 1
            x = np.array(x)
            hist = np.histogram(x, bins=bin_edges)
            samples.append(Sample([hist[1]], hist[0], title=f'{sample_title}_itoy{i}', analysis_type=analysis_type, sample_title=sample_title))
    else: #in case we do not have many toys
        x = np.array(xvar[0])
        hist =hist = np.histogram(x, bins=bin_edges)
        samples.append(Sample([hist[1]], hist[0], title=f'{sample_title}', analysis_type=analysis_type, sample_title=sample_title))
    return samples

def bin_sample2D(xvar, yvar, sample_title=None, analysis_type=None, bin_edges=None):
    """Bins a toy in 2D histogram

    Parameters
    ----------
    title : string
        The title of the sample
    xvar : numpy.array
         The array of values of the observable on the x axis
    yvar : numpy.array
         The array of values of the observable on the y axis
    analysis_type : 
        Binning type for this sample
    Returns
    ------
    Sample2D
        The object of Sample2D class binning (xvar, yvar) array
    """
    if bin_edges is None and analysis_type is not None:
        bin_xedges = analysis_type_xedges[analysis_type]
        bin_yedges = analysis_type_yedges[analysis_type]  
    elif bin_edges is None and analysis_type is None:
        raise ValueError("Either 'bin_edges' or 'analysis_type' must be provided.")

    
    samples = []
    i = -1
    if len(xvar) > 1:
        for x, y in zip(xvar, yvar): 
            i += 1
            x, y = np.array(x), np.array(y)
            hist = np.histogram2d(x, y, bins=(bin_xedges, bin_yedges))
            samples.append(Sample([hist[1], hist[2]], hist[0], title=f'{sample_title}_itoy{i}', analysis_type=analysis_type, sample_title=sample_title))
    else: #in case we do not have many toys
        x, y = np.array(xvar[0]), np.array(yvar[0])
        hist = np.histogram2d(x, y, bins=(bin_xedges, bin_yedges))
        samples.append(Sample([hist[1], hist[2]], hist[0], title=f'{sample_title}', analysis_type=analysis_type, sample_title=sample_title))
    return samples
        

def calculate_dchi2(sample_obs, sample_exp, perbin=False):
    """Calculates a -2deltalnL for given observed and expected samples distributions per bin of the observables or as sum over all bins

    Parameters
    ----------
    sample_obs : Sample1D or Sample2D
        Contains observed events distribution 
    sample_exp : Sample1D or Sample2D
        Contains expected events distribution 
    perbin : bool
        True - calculate -2deltalnL for each bin separetely
        False - calculate total -2deltalnL 
    Returns
    ------
    Float or numpy.array
        Value/values of -2deltalnL
    """
    valid_mask = (sample_obs.z > 0) & (sample_exp.z > 0)
    result = np.zeros_like(sample_obs.z, dtype=float)
    invalid_mask = ~valid_mask
    result[valid_mask] = sample_exp.z[valid_mask]-sample_obs.z[valid_mask] - sample_obs.z[valid_mask]* \
                                np.log(sample_exp.z[valid_mask]/sample_obs.z[valid_mask]) 
    result[invalid_mask] = sample_exp.z[invalid_mask]-sample_obs.z[invalid_mask]
    if not perbin:
        result = np.sum(result)
    return 2*result

def project_all_samples(toy, axis, verbose=False):
    """Projects all samples in a ToyXp object along a specified axis.

    Parameters
    ----------
    toy : ToyXp
        The ToyXp object containing the samples to be projected.
    axis : string
        The axis to project along. Should be one of 'energy' or 'angle'. Here 'energy' stand for both Erec and p
    verbose : bool, optional
        If True, prints messages for 1D samples that are skipped. Default is False.

    Returns
    -------
    ToyXp
        A new ToyXp object containing the projected 1D samples.
    
    Raises
    ------
    ValueError
        If an invalid value for 'axis' is provided (other than 'energy' or 'angle').
    """
    toy_1D = ToyXp()
    for sample in toy.samples:
        if sample.ndim() > 1:
            if axis == 'energy':
                sample_proj = sample.project_to_x()
            elif axis == 'angle':
                sample_proj = sample.project_to_y()
            else:
                raise ValueError(f'Invalid axis value: {axis}. Values "energy" or "angle" are only allowed.')
        else:
            sample_proj = None
        if sample.ndim() == 1 and verbose:
            print(f'Sample {sample.title} is already 1D. Skipped')
            sample_proj = None

        if sample.ndim() == 1 and ((axis=='energy' and (analysis_type_to_energyvar[sample.analysis_type]=='Erec' or analysis_type_to_energyvar[sample.analysis_type]=='p')) or (axis=='angle' and (analysis_type_to_anglevar[sample.analysis_type]=='Theta'))):
            sample_proj = sample
        
        if sample_proj:
            toy_1D.append(sample_proj)
        
    return toy_1D
        

def merge_for_inter_plotting(toy):
    """Merges samples the flavour and nominal interaction modes components to config_samples.interaction_modes_2 for plotting.

    Parameters
    ----------
    toy : ToyXp
        The ToyXp object containing samples from various interaction modes and oscillation channels.

    Returns
    -------
    ToyXp
        A new ToyXp object containing the merged samples for each interaction mode, 
        ready for plotting. Each merged sample corresponds to a unique interaction mode 2.
    """
    toy_per_int_mode = ToyXp()
    
    for sample_title in toy.sample_titles:
        sample_per_int = defaultdict(lambda: 0)
        for inter in interaction_modes:
            for flavour in osc_channels:
                title = f'{sample_title}_{inter}_{flavour}'
                sample_per_int[inter_to_inter2[inter]] =  sample_per_int[inter_to_inter2[inter]] + toy[title]

        for inter2 in interaction_modes_2:
            sample_per_int[inter2].set_title(f'{sample_title}_{inter2}')
            sample_per_int[inter2].set_sample_title(f'{sample_title}')
            toy_per_int_mode.append(sample_per_int[inter2])
            
    return toy_per_int_mode

def merge_for_flavour_plotting(toy):
    """Merges the samples osc.channels component to have only break down by flavours for plotting.

    Parameters
    ----------
    toy : ToyXp
        The ToyXp object containing samples from various interaction modes and flavours.

    Returns
    -------
    ToyXp
        A new ToyXp object containing the merged samples for each flavour, 
        ready for plotting. Each merged sample corresponds to a unique flavour.
    """

    toy_per_flavour = ToyXp()
    
    for sample_title in toy.sample_titles:
        sample_per_flavour = defaultdict(lambda: 0)
        for osc_channel in osc_channels :
            for inter in interaction_modes:
                title = f'{sample_title}_{inter}_{osc_channel}'
                sample_per_flavour[osc_channel] =  sample_per_flavour[osc_channel] + toy[title]

        for osc_channel in osc_channels:
            sample_per_flavour[osc_channel].set_title(f'{sample_title}_{osc_channel}')
            sample_per_flavour[osc_channel].set_sample_title(f'{sample_title}')
            toy_per_flavour.append(sample_per_flavour[osc_channel])
            
    return toy_per_flavour


class UnbinnedSample:
    def __init__(self, xvar, yvar=None, title=None, analysis_type=None, sample_title=None):
        """
        Initializes a UnbinnedSample class that can handle both 1D and 2D unbinned samples.
    
        Parameters
        ----------
        xvar : array-like
            Contains the x-axis values of the sample.
        yvar : array-like, optional
            Contains the y-axis values for 2D samples.
        title : string, optional
            Title of the sample.
        analysis_type : string, optional
            The type of the binning, defining the analysis to be performed.
        """
        self.__title = title
        self.__xvar = xvar
        self.__yvar = yvar
        self.__analysis_type = analysis_type
        if sample_title is None:
            sample_title = title
        self.__sample_title = sample_title

        if yvar and len(xvar) != len(yvar):
            ValueError (f'yvar and xvar should have the same length, however len(yvar)={len(xvar)}, len(xvar)={len(yavr)}')

    def __str__(self):
        """
        Returns a string representation of the Sample object.
        
        Returns
        -------
        string
            A string containing the title, dimensions, and analysis type of the sample.
        """
        return f"Title: {self.title}; Sample title: {self.sample_title}; Dimension: {self.ndim()}; Analysis type: {self.analysis_type} Integral: {self.contsum()}"
    
    @property
    def title(self):
        """
        Getter for the title of the sample.

        Returns
        -------
        string
            The title of the sample.
        """
        return self.__title
        
    @property
    def sample_title(self):
        """
        Getter for the sample_title of the sample.

        Returns
        -------
        string
            The title of the sample.
        """
        return self.__sample_title

    @property
    def xvar(self):
        """
        Getter for the x-variable of the sample.

        Returns
        -------
        array-like
            The x-variable (e.g., xedges) of the sample.
        """
        return self.__xvar

    @property
    def yvar(self):
        """
        Getter for the y-variable of the sample (if present).
        
        Returns
        -------
        array-like, optional
            The y-variable (e.g., yedges) of the sample if available; otherwise None.
        """
        return self.__yvar

    @property
    def analysis_type(self):
        """
        Getter for the analysis type of the sample.
        
        Returns
        -------
        string, optional
            The type of the analysis (e.g., binning scheme).
        """
        return self.__analysis_type
    
    def ndim(self):
        """
        Returns the number of dimensions of the sample.
        
        Returns
        -------
        int
            1 for 1D samples, 2 for 2D samples.
        """
        if self.__yvar is not None:
            return 2
        return 1

    def contsum(self):
        """
        Returns the number of dimensions of the sample.
        
        Returns
        -------
        int
            1 for 1D samples, 2 for 2D samples.
        """
        return len(self.__xvar) 
        
    def plot(self, ax, wtitle=True, wtag=False, **kwargs):
        """
        Plots the distribution of the UnbinnedSample object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object where the plot will be drawn.
        wtitle : bool, optional
            If True, the title of the sample will be shown on the plot. Default is True.
        wtag : bool, optional
            If True, a tag will be displayed on the plot. Default is False.
        **kwargs : additional keyword arguments
            Additional arguments passed to the plotting function.
        
        Raises
        ------
        ValueError
            If the sample has more than 2 dimensions.
        """
        if self.ndim() == 1:
            self._plot_1d(ax, wtitle, wtag, **kwargs)
        elif self.ndim() == 2:
            self._plot_2d(ax, wtitle, wtag, **kwargs)
        else:
            raise ValueError("Plotting is only supported for 1D and 2D samples.")    
    
    def _plot_2d(self, ax, wtitle, wtag, **kwargs):
        """
        Plots a 2D sample using a scatter plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object where the plot will be drawn.
        wtitle : bool, optional
            If True, the title will be displayed on the left of the plot. Default is True.
        wtag : bool, optional
            If True, a tag will be displayed on the right of the plot. Default is False.
        **kwargs : additional keyword arguments
            Additional arguments passed to the scatter plot function.
        """
        ax.scatter(self.__xvar, self.__yvar, facecolors='white', edgecolors='black', s=20, **kwargs)
        
        if self.title is not None and wtitle:
            ax.set_title(sample_to_title[self.title], loc='left')
        if self.analysis_type is not None:
            _=ax.set_xticks(analysis_type_to_xtickspos[self.analysis_type])
            ax.set_xlim(0.001, analysis_type_to_xmax[self.analysis_type])
            ax.set_xlabel(analysis_type_to_xlabel[self.analysis_type])
        if wtag:
            ax.set_title(tag, loc='right')
    
        _=ax.set_yticks([30*i for i in range(7)])                          
        ax.set_ylim(0, 180)
        ax.set_ylabel("Angle [degrees]")
        if wtag:
            ax.set_title(tag, loc='right', fontsize=20)
            
    def _plot_1d(self, ax, wtitle, wtag, **kwargs):
        """
        Plots a binned distribution of the 1D sample.

        Since unbinned 1D data doesn't make sense to plot, the data is binned and plotted instead.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object where the plot will be drawn.
        wtitle : bool, optional
            If True, the title will be displayed on the plot. Default is True.
        wtag : bool, optional
            If True, a tag will be displayed on the plot. Default is False.
        **kwargs : additional keyword arguments
            Additional arguments passed to the plot function.
        """
        binned_sample_list = bin_sample1D(self.__xvar, sample_title=self.__sample_title, analysis_type=self.__analysis_type,) #the result of bin_sample1D is a list
        binned_sample_list[0].plot(ax, wtitle, wtag, kind='data', **kwargs)

        
class Sample:
    """
    Represents a histogram sample that can handle both 1D and 2D distributions.
    The class provides functionalities for slicing, rebinning, plotting, and 
    other operations on the histogram data.

    Attributes
    ----------
    title : str
        The title of the sample.
    bin_edges : list of numpy.ndarray
        A list containing arrays of bin edges. For a 1D histogram, it contains 
        one array for the x-axis edges. For a 2D histogram, it contains two arrays 
        for the x-axis and y-axis edges.
    z : numpy.ndarray
        The histogram values, stored in a 1D or 2D array depending on the number 
        of dimensions.
    analysis_type : str
        The type of analysis or binning used (e.g., 'Erec', 'e-theta', or 'PTheta').

    Methods
    -------
    ndim() : int
        The number of dimensions of the histogram (1 for 1D, 2 for 2D).
    contsum():
        Returns the total sum of bin contents of the histogram.
    plot(ax, wtag, **kwargs):
        Plots the histogram distribution on a given matplotlib axis.
    rebin(new_bin_edges):
        Rebins the histogram using the specified new bin edges.
    slice(*args):
        Returns a sliced portion of the histogram based on specified binning edges.
    """
    def __init__(self, bin_edges, z, title=None, analysis_type=None, sample_title=None):
        """
        Initializes a Sample class that can handle both 1D and 2D histograms.
    
        Parameters:
        ----------
        title : string
            Title of the sample
        bin_edges: List
            Contains bin edges, e.g., [xedges] for 1D or [xedges, yedges] for 2D
        z: array-like
            Contains histogram values, 1D or 2D depending on the number of dimensions
        analysis_type : string
            The type of the binning
        """
        self.__title = title
        self.__bin_edges = bin_edges
        self.__z = np.array(z)
        self.__analysis_type = analysis_type
        if sample_title is None:
            sample_title = title
        self.__sample_title = sample_title


        if len(bin_edges) != np.ndim(z):
            raise ValueError(
                f"Dimension mismatch: bin_edges has {len(bin_edges)} dimensions, "
                f"but z has {np.ndim(z)} dimensions.")
            
        expected_shape = tuple(len(edges) - 1 for edges in bin_edges)
        bin_egdes_shape = tuple(len(edges) for edges in bin_edges)

        if expected_shape != self.__z.shape:
            raise ValueError(
                f"Shape mismatch: bin_edges implies shape {bin_egdes_shape}, "
                f"but z has shape {self.__z.shape}. Numer of bin edges = number of bins + 1")
    
    def __str__(self):
        return f"Title: {self.title}; Sample title: {self.sample_title}; Dimension: {self.ndim()}; Shape: {self.z.shape}; Analysis type: {self.analysis_type} Integral: {self.contsum()}"
    
    def __add__(self, other): 
        if isinstance(other, Sample) and self._check_bin_egdes_matching(other):
            return Sample(self.bin_edges, self.z + other.z, self.title, self.analysis_type, self.sample_title)
        elif isinstance(other,(int, float, np.ndarray)):
            return Sample(self.bin_edges, self.z + other, self.title, self.analysis_type, self.sample_title)
        raise TypeError("Incompatible operand type or size")

    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        if isinstance(other, Sample) and self._check_bin_egdes_matching(other):
            return Sample(self.bin_edges, self.z - other.z, self.title, self.analysis_type, self.sample_title)
        elif isinstance(other,(int, float, np.ndarray)):
            return Sample(self.bin_edges, self.z - other, self.title, self.analysis_type, self.sample_title)
        raise TypeError("Incompatible operand type or size")
        
    def __rsub__(self, other):
        return self.__sub__(other)
        
    def __mul__(self, other):
        if (isinstance(other, Sample) and self.bin_edges == other.bin_edges):
            return Sample(self.bin_edges, self.z * other.z)
        elif isinstance(other,(int, float)):
            return Sample(self.bin_edges, self.z * other, self.title, self.analysis_type, self.sample_title)
        raise ValueError("Incompatible operand type or size")

    def __rmul__(self, other):
        return self.__mul__(other)
            
    def __neg__(self):
        return Sample(self.bin_edges, -self.z, self.title, self.analysis_type)

    def __truediv__(self, other):
        
        ifbinedgesmatch = self._check_bin_egdes_matching(other)
        
        if isinstance(other, Sample) and ifbinedgesmatch:
            ratio = divide_arrays(self.z, other.z)
            return Sample(self.bin_edges, ratio, self.title, self.analysis_type, self.sample_title)
        raise ValueError("Incompatible operand type or size")
        
    
    #Set getters
    @property
    def title(self):
        return self.__title
    
    @property
    def sample_title(self):
        return self.__sample_title

    @property
    def bin_edges(self):
        return self.__bin_edges

    @property
    def z(self):
        return self.__z

    @property
    def analysis_type(self):
        return self.__analysis_type
    
    @property
    def inter_mode(self):
        return self.__inter_mode
    
    @property
    def flavour(self):
        return self.__flavour    
    
    #Set setters
    def set_title(self, title):
        self.__title = title
    def set_sample_title(self, sample_title):
        self.__sample_title = sample_title
    def set_z(self, z):
        self.__z = z
    
    #Set methods
    def ndim(self):
        return len(self.__bin_edges)

    def contsum(self):
        return np.sum(self.__z)

    def bin_centers(self):
        return [(bin_edges[1:] + bin_edges[:-1])/2 for bin_edges in self.bin_edges]

    def plot(self, ax, wtitle=True, wtag=False, kind='hist', yerr=None, show_colorbar=True, rotate=False, **kwargs):
        """
        Plots the distrubution of the Sample object.

        Parameters:
        ----------
        ax : string
            Title of the sample
        wtag : bool
            Set True (False) to (not) show the tag 
        """
        plot = None
        if self.ndim() == 1:
            self._plot_1d(ax, wtitle, wtag, kind, yerr, rotate, **kwargs)
        elif self.ndim() == 2:
            plot = self._plot_2d(ax, wtitle, wtag, kind, show_colorbar, **kwargs)
        else:
            raise ValueError("Plotting is only supported for 1D and 2D samples.")   
        return plot

    def rebin(self, new_bin_edges):
        """
        Rebins the distribution of the `Sample` object using the specified 
        new binning edges. Supports rebinning for both 1D and 2D distributions.
    
        Parameters
        ----------
        new_bin_edges : list of arrays
            The new binning edges to use for rebinning the distribution. 
            For a 1D distribution, provide a single array of bin edges. 
            For a 2D distribution, provide a list containing two arrays: 
            the bin edges along the x-axis and y-axis.
    
        Returns
        -------
        Sample
            A new `Sample` object with the distribution rebinned according 
            to the specified bin edges.
    
        Raises
        ------
        ValueError
            If the distribution is not 1D or 2D, as rebinning is only 
            supported for these cases.
    
        Notes
        -----
        - The new binning edges should fully encompass the original 
          distribution's range for meaningful results.
        """
        if self.ndim() == 1:
            return self._rebin_1d(new_bin_edges)
        elif self.ndim() == 2:
            return self._rebin_2d(new_bin_edges)
        else:
            raise ValueError("Rebinning is only supported for 1D and 2D samples.")
            
    def project_to_x(self):
        if self.ndim()>1:
            return self.rebin([self.bin_edges[0], [self.bin_edges[1][0], self.bin_edges[1][-1]]])
        print('Trying to project 1D. The sample will not be modified.')
        return self  
    
    def project_to_y(self):
        if self.ndim()>1:
            return self.rebin([[self.bin_edges[0][0], self.bin_edges[0][-1]], self.bin_edges[1]])
        raise ValueError('Trying to project 1D')
    
    def slice(self, *args):
        """
        Returns a sliced portion of the distribution for the `Sample` object, 
        based on specified binning edges. The method supports slicing both 
        one-dimensional and two-dimensional distributions.
    
        Parameters
        ----------
        *args : float
            For 1D distributions, provide two arguments: `xmin` and `xmax`, 
            representing the lower and upper edges of the slice, respectively.
            For 2D distributions, provide four arguments: `xmin`, `xmax`, 
            `ymin`, and `ymax`, representing the lower and upper edges 
            of the slice in both dimensions.
    
        Returns
        -------
        Sample
            A new `Sample` object representing the sliced portion of the 
            original distribution, with updated binning edges and data.
    
        Raises
        ------
        ValueError
            If the specified slice edges are not contained within the existing 
            binning edges.
    
        Notes
        -----
        - The binning edges provided for slicing must align with the existing 
          binning edges of the `Sample` object.
        - In the 1D case, the slicing is performed along the x-axis.
        - In the 2D case, the slicing is performed along both the x and y axes. 
        """
        if self.ndim() == 1:
            xmin, xmax = args
            if not(any(np.isclose(xmin, self.bin_edges[0])) and any(np.isclose(xmax, self.bin_edges[0]))):
                raise ValueError("The slice edges should be contained in the binning edges")
            start = np.where(np.isclose(self.bin_edges[0], xmin))[0][0]
            stop = np.where(np.isclose(self.bin_edges[0], xmax))[0][0]
            new_xedges = self.bin_edges[0][start:stop + 1]
            new_z = self.z[start:stop]
            return Sample(title=self.title, bin_edges=[new_xedges], z=new_z, analysis_type=self.analysis_type)

        elif self.ndim() == 2:
            xmin, xmax, ymin, ymax = args
            if not(xmin in self.bin_edges[0] and xmax in self.bin_edges[0] and ymin in self.bin_edges[1] and ymax in self.bin_edges[1]):
                raise ValueError("The slice edges should be contained in the binning edges")
            start_x = np.where(self.bin_edges[0] == xmin)[0][0]
            stop_x = np.where(self.bin_edges[0] == xmax)[0][0]
            start_y = np.where(self.bin_edges[1] == ymin)[0][0]
            stop_y = np.where(self.bin_edges[1] == ymax)[0][0]
            new_xedges = self.bin_edges[0][start_x:stop_x + 1]
            new_yedges = self.bin_edges[1][start_y:stop_y + 1]
            new_z = self.z[start_x:stop_x, start_y:stop_y]
            return Sample(title=self.title, bin_edges=[new_xedges, new_yedges], z=new_z, analysis_type=self.analysis_type)

    def _plot_1d(self, ax, wtitle, wtag, kind, yerr, rotate, **kwargs):
        
        if kind == 'hist':
            plot_histogram(ax, self.bin_edges[0], self.z, rotate, **kwargs)
        else:
            plot_data(ax, self.bin_edges[0], self.z, yerr, **kwargs)

        if self.sample_title is not None and wtitle:
            ax.set_title(sample_to_title[self.sample_title], loc='left')
        if self.analysis_type is not None:
            _=ax.set_xticks(analysis_type_to_xtickspos[self.analysis_type])
            ax.set_xlim(0.001, analysis_type_to_xmax[self.analysis_type])
            ax.set_xlabel(analysis_type_to_xlabel[self.analysis_type])
        if wtag:
            ax.set_title(tag, loc='right')
    
        ax.autoscale(axis='y', tight=False)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Number of events')

    def _plot_2d(self, ax, wtitle, wtag, kind, show_colorbar, **kwargs):
        
        label = kwargs.pop('label', None)
        if self.title is not None and wtitle:
            ax.set_title(sample_to_title[self.sample_title], loc='left', fontsize=20)
        if self.analysis_type is not None:
            _=ax.set_xticks(analysis_type_to_xtickspos[self.analysis_type])
            ax.set_xlim(0.001, analysis_type_to_xmax[self.analysis_type])
            ax.set_xlabel(analysis_type_to_xlabel[self.analysis_type])

        _=ax.set_yticks([30*i for i in range(7)])                          
        ax.set_ylim(0, 180)
        ax.set_ylabel("Angle, [degrees]")
        if wtag:
            ax.set_title(tag, loc='right', fontsize=20)
            
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x_mask = (self.bin_edges[0][:-1] >= xlim[0]) & (self.bin_edges[0][1:] <= xlim[1])
        y_mask = (self.bin_edges[1][:-1] >= ylim[0]) & (self.bin_edges[1][1:] <= ylim[1])
        displayed_z = self.z[np.ix_(x_mask, y_mask)]
        vmax = displayed_z.max() if displayed_z.size > 0 else None
        mesh = ax.pcolormesh(self.bin_edges[0], self.bin_edges[1], self.z.transpose(), zorder=0, 
                             cmap=kwargs.pop('cmap', rev_afmhot), vmax=vmax,  **kwargs)
        if show_colorbar:
            cbar = plt.colorbar(mesh, ax=ax)
        
        #ax.pcolormesh does not support labels. Dummy patches are introduced instead
        if label:
            rectangle = mpatches.Rectangle((100.1, 0.1), 0.3, 0.2, linewidth=1, edgecolor=None, facecolor='orange', label=label)
            ax.add_patch(rectangle)
        return mesh
    
    def _rebin_1d(self, new_bin_edges):
        if len(new_bin_edges) != 1:
            raise ValueError(f"The dimension of new edges ({len(new_bin_edges)}) should be equal to 1 as it is 1D sample")
            
        new_xedges = np.round(new_bin_edges[0], 6)
        old_xedges = np.round(self.bin_edges[0], 6)
        
        if all(new_xedge in old_xedges for new_xedge in new_xedges):
            i = 0
            new_z = np.zeros(new_xedges.shape[0]-1)
            for k in range(self.z.shape[0]):
                if old_xedges[k] >= new_xedges[i+1]:
                    i += 1
                new_z[i] += self.z[k]
            print(self.analysis_type)
            return Sample([new_xedges], new_z, self.title, self.analysis_type, self.sample_title)
        else:
            raise ValueError("New edges should be fully contained in old edges")


    def _rebin_2d(self, new_bin_edges):
        if len(new_bin_edges) != 2:
            raise ValueError(f"The dimension of new edges ({len(new_bin_edges)}) should be equal to 2 as it is 2D sample")
        new_xedges = new_bin_edges[0]
        new_yedges = new_bin_edges[1]
        old_xedges = self.bin_edges[0]
        old_yedges = self.bin_edges[1]
        
        if all(new_xedge in old_xedges for new_xedge in new_xedges) and all(new_yedge in old_yedges for new_yedge in new_yedges):
            new_z = np.zeros((len(new_xedges)-1, len(new_yedges)-1))
            j = 0
            for m in range(self.z.shape[1]):
                if old_yedges[m] >= new_yedges[j+1]:
                    j+=1
                i = 0
                for k in range(self.z.shape[0]):
                    if old_xedges[k] >= new_xedges[i+1]:
                        i += 1
                    new_z[i][j] += self.z[k][m]
                    
            if len(new_yedges) > 2 and len(new_xedges) > 2:
                return Sample( [new_xedges, new_yedges], new_z, self.title, self.analysis_type, self.sample_title)
            if len(new_yedges) == 2:
                if self.analysis_type=='e-theta':
                    analysis_type = 'Erec'
                elif self.analysis_type=='PTheta' or self.analysis_type=='p-theta':
                    analysis_type = 'P'
                else:
                    analysis_type = None
                return Sample( [new_xedges], new_z.transpose()[0], self.title, analysis_type, self.sample_title)
            if len(new_xedges) == 2:
                if self.analysis_type=='e-theta' or self.analysis_type=='PTheta' or self.analysis_type=='p-theta':
                    analysis_type = 'Theta'
                else:
                    analysis_type = None
                return Sample( [new_yedges], new_z[0], self.title, analysis_type, self.sample_title)       
        else:
            raise ValueError("New edges should be contained fully in old edges")
            
    def _check_bin_egdes_matching(self, other):

        ifbinedgesmatch = np.allclose(self.bin_edges[0], other.bin_edges[0])
        if self.ndim() == 2:
            ifbinedgesmatch = ifbinedgesmatch and np.allclose(self.bin_edges[1], other.bin_edges[1])
        return ifbinedgesmatch

class ToyXp:
    """
    A class representing a collection of samples, which can be accessed,
    modified, and managed collectively.

    Attributes
    ----------
    samples : list
        A list of `Sample` objects representing the samples in the collection.

    Methods
    -------
    get_titles():
        Returns the titles of all samples in the collection.
    get_sample(title):
        Retrieves a sample from the collection by its title.
    append(sample):
        Adds a new sample to the collection.
    """

    def __init__(self):
        """
        Initializes the `ToyXp` object with a list of samples.

        Parameters
        ----------
        samples_list : list
            A list of `Sample` objects to initialize the collection.
        """
        self.samples = []

    def __str__(self):
        result = f'{len(self.samples)} samples are included in this toy: \n'
        result += '\n'.join(str(sample) for sample in self.samples)
        return result
           
    def __getitem__(self, key):
        for index, sample in enumerate(self.samples):
            if sample.title == key:
                return self.samples[index]
        raise ValueError(f"Sample with title {key} not found")

    @property
    def titles(self):
        """
        Returns the titles of all samples in the collection.

        Returns
        -------
        list
            A list of titles for all `Sample` objects in the collection.
        """
        titles = []
        for sample in self.samples:
            titles.append(sample.title)
        return titles
    
    @property
    def sample_titles(self):
        """
        Returns the titles of all samples in the collection.

        Returns
        -------
        list
            A list of titles for all `Sample` objects in the collection.
        """
        sample_titles = []
        for sample in self.samples:
            sample_titles.append(sample.sample_title)
        return list(set(sample_titles))

    def append(self, sample):
        """
        Adds a new sample to the collection.

        Parameters
        ----------
        sample : Sample
            The `Sample` object to add to the collection.
        """
        if isinstance(sample, Sample) or isinstance(sample, UnbinnedSample):
            self.samples.append(sample)
        else:
            raise ValueError('The element of ToyXp should be Sample object')

    def filter_by_sample_title(self, sample_title):
        """
        Filters samples by their sample title and returns a ToyXp object with the matching samples.

        Parameters
        ----------
        sample_title : string
            The title of the sample to filter by.

        Returns
        -------
        ToyXp
            A new ToyXp object containing samples that match the provided sample_title.
        """
        sample_per_int_mode = ToyXp()
        for sample in self.samples:
            if sample.sample_title == sample_title:
                sample_per_int_mode.append(sample)
        return sample_per_int_mode

    def project_to_x(self):
        """
        Projects all samples to the x-axis and returns a new ToyXp object containing the projected samples.

        This method assumes that each sample has a `project_to_x` method that performs the projection.

        Returns
        -------
        ToyXp
            A new ToyXp object containing the projected samples along the x-axis.
        """
        projected_toy = ToyXp()
        for sample in self.samples:
            projected_toy.append(sample.project_to_x())
        return projected_toy

    def project_to_y(self):
        """
        Projects all samples to the y-axis and returns a new ToyXp object containing the projected samples.

        This method assumes that each sample has a `project_to_y` method that performs the projection.

        Returns
        -------
        ToyXp
            A new ToyXp object containing the projected samples along the y-axis.
        """
        prejected_toy = ToyXp()
        for sample in self.samples:
            prejected_toy.append(sample.project_to_y())
        return prejected_toy

 


            
            

