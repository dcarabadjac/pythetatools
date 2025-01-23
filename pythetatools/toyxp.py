import numpy as np
from matplotlib import pyplot as plt
from .global_names import *
import subprocess
from array import array
import os
import uproot
from .file_manager import download
from .base_visualisation import *

    

def download_toyxp(input_path, login=my_login, domain=my_domain):
    """Dowloads the PTGenerateXp output root file 
    
    Parameters
    ----------
    filename : string
        The filename of PTGenerateXp output root file
    """
    download(input_path, login=login, domain=domain, destination=os.path.join(inputs_dir, 'ToyXp'))
    

def get_samples_info(filename):
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
    
    with uproot.open(filename) as file:
        keys = file.keys() # Get list of all keys        
        for key in keys:
            if (file[key].classname.startswith("TH2") or file[key].classname.startswith("TH1")) and not ('_true' in key): #select only rec distr
                analysis_type_key = f"{key[1:-2]}_analysis_type" # Get the corresponding analysis type for this sample
                analysis_type = file[analysis_type_key].all_members['fTitle']
                samples_dict[key[1:-2]] = analysis_type
    return samples_dict



def load_hists(filename):
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
    
    samples_dict = get_samples_info(filename)
    toy = ToyXp()
    
    with uproot.open(filename) as file:
        for sample_title, analysis_type in samples_dict.items():
            hist = file[f'h{sample_title}']
            xedges = hist.axis(0).edges()
            z = hist.values()
            if analysis_type =='e-theta' or analysis_type =='PTheta':
                yedges = hist.axis(1).edges()
                toy.append(Sample([xedges, yedges], z, sample_title, analysis_type))    
            else:
                toy.append(Sample([xedges], z, sample_title, analysis_type))  
        
    return toy

def load_tree(filename, sample_title, analysis_type, itoy):
    """Loads a unbinned toy for a given sample

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
        input_tree = file.get("ToyXp")
        
        if input_tree is None:
            print(f"'ToyXp' tree not found in file {filename}")
            return None

        if analysis_type =='e-theta' or analysis_type =='PTheta':
            xvar_branch_name = f"{analysis_type_to_xvar[analysis_type]}_{sample_title}"
            t_branch_name = f"t_{sample_title}"
            data = input_tree.arrays([xvar_branch_name, t_branch_name], entry_start=itoy, entry_stop=itoy + 1)
            xvar = data[xvar_branch_name][0]  
            theta = data[t_branch_name][0]
            return (np.array(xvar), np.array(theta))

        xvar_branch_name = f"{analysis_type_to_xvar[analysis_type]}_{sample_title}"
        data = input_tree.arrays([xvar_branch_name], entry_start=itoy, entry_stop=itoy + 1)
        xvar = data[xvar_branch_name][0]            
        return (np.array(xvar), )

def load_toy(filename, itoy):
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
    samples_dict = get_samples_info(filename)
    toy = ToyXp()
    with uproot.open(filename) as file:
        for sample_title, analysis_type in samples_dict.items():
            if analysis_type_to_dim[analysis_type] == '1D':
                sample = bin_sample1D(sample_title, *load_tree(filename, sample_title, analysis_type, itoy), analysis_type)
                toy.append(sample)
            elif analysis_type_to_dim[analysis_type] == '2D':
                sample = bin_sample2D(sample_title, *load_tree(filename, sample_title, analysis_type, itoy), analysis_type)
                toy.append(sample)
    return toy

def load(filename, type, itoy=None):
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
    
    toy = ToyXp() #
    if type=='asimov':
        toy = load_hists(filename)
    elif type=='toy':
        toy = load_toy(filename, itoy)

    return toy

def bin_sample1D(title, xvar, analysis_type):
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
    
    hist = np.histogram(xvar, bins=analysis_type_xedges[analysis_type])
    return Sample(title, [hist[1]], hist[0], analysis_type)

def bin_sample2D(title, xvar, yvar, analysis_type):
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
    
    hist = np.histogram2d(xvar, yvar, bins=[analysis_type_xedges[analysis_type], analysis_type_yedges[analysis_type]])
    return Sample(title, [hist[1], hist[2]], hist[0], analysis_type)


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
    zero_mask = (sample_obs.z == 0)
    result = np.zeros_like(sample_obs.z, dtype=float)
    non_zero_mask = ~zero_mask
    result[non_zero_mask] = sample_exp.z[non_zero_mask]-sample_obs.z[non_zero_mask] - sample_obs.z[non_zero_mask]* \
                                np.log(sample_exp.z[non_zero_mask]/sample_obs.z[non_zero_mask]) 
    result[zero_mask] = sample_exp.z[zero_mask]
    if not perbin:
        result = np.sum(result)
    return result



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
    ndim : int
        The number of dimensions of the histogram (1 for 1D, 2 for 2D).

    Methods
    -------
    nevents():
        Returns the total number of events in the histogram.
    plot(ax, wtag, **kwargs):
        Plots the histogram distribution on a given matplotlib axis.
    rebin(new_bin_edges):
        Rebins the histogram using the specified new bin edges.
    slice(*args):
        Returns a sliced portion of the histogram based on specified binning edges.
    """
    def __init__(self, bin_edges, z, title=None, analysis_type=None):
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

        if len(bin_edges) != np.ndim(z):
            raise ValueError(
                f"Dimension mismatch: bin_edges has {len(bin_edges)} dimensions, "
                f"but z has {np.ndim(z)} dimensions."
            )
        expected_shape = tuple(len(edges) - 1 for edges in bin_edges)
        bin_egdes_shape = tuple(len(edges) for edges in bin_edges)

        if expected_shape != self.__z.shape:
            raise ValueError(
                f"Shape mismatch: bin_edges implies shape {bin_egdes_shape}, "
                f"but z has shape {self.__z.shape}. Numer of bin edges = number of bins + 1"
            )
            
    #Implement useful special methods
    def __str__(self):
        return f"Sample: {self.title}; Dimension: {self.ndim()}; Shape: {self.z.shape} Total events: {self.nevents()}"
    
    def __add__(self, other):
        if isinstance(other, Sample) and self.bin_edges == other.bin_edges:
            return Sample(self.bin_edges, self.z + other.z)
        raise ValueError("Incompatible types or sizes of operands")
        
    def __sub__(self, other):
        if isinstance(other, Sample) and self.bin_edges == other.bin_edges:
            return Sample(self.bin_edges, self.z - other.z)
        raise ValueError("Incompatible types of operands")
        
    def __mul__(self, other):
        if isinstance(other, Sample) and self.bin_edges == other.bin_edges:
            return Sample(self.bin_edges, self.z * other.z)
        raise ValueError("Incompatible types of operands")
            
    def __neg__(self):
        return Sample(self.bin_edges, -self.z)

    def __truediv__(self, other):
        if isinstance(other, Sample) and self.bin_edges == other.bin_edges:
            where_zeros = other.z==0
            ratio = np.zeros_like(self.z)
            ratio[where_zeros] = 0
            ratio[~where_zeros] = self.z[~where_zeros] / other.z[~where_zeros]
            return Sample(self.bin_edges, ratio)
        raise ValueError("Incompatible types of operands")
        

    #Set getters
    @property
    def title(self):
        return self.__title

    @property
    def bin_edges(self):
        return self.__bin_edges

    @property
    def z(self):
        return self.__z

    @property
    def analysis_type(self):
        return self.__analysis_type

    def ndim(self):
        return len(self.__bin_edges)

    def nevents(self):
        return np.sum(self.__z)

    def plot(self, ax, wtag=False, **kwargs):
        """
        Plots the distrubution of the Sample object.

        Parameters:
        ----------
        ax : string
            Title of the sample
        wtag : bool
            Set True (False) to (not) show the tag 
        """
        if self.ndim() == 1:
            self._plot_1d(ax, wtag, **kwargs)
        elif self.ndim() == 2:
            self._plot_2d(ax, wtag, **kwargs)
        else:
            raise ValueError("Plotting is only supported for 1D and 2D samples.")     

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
        return self.rebin([self.bin_edges[0], [self.bin_edges[1][0], self.bin_edges[1][-1]]])
    
    def project_to_y(self):
        return self.rebin([[self.bin_edges[0][0], self.bin_edges[0][-1]], self.bin_edges[1]])

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
            if not(xmin in self.bin_edges[0] and xmax in self.bin_edges[0]):
                raise ValueError("The slice edges should be contained in the binning edges")
            start = np.where(self.bin_edges[0] == xmin)[0][0]
            stop = np.where(self.bin_edges[0] == xmax)[0][0]
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

    def _plot_1d(self, ax, wtag=False, **kwargs):
        plot_histogram(ax, self.bin_edges[0], self.z, **kwargs)

        if self.title is not None:
            ax.set_title(sample_to_title[self.title], loc='left')
        if self.analysis_type is not None:
            _=ax.set_xticks(analysis_type_to_xtickspos[self.analysis_type])
            ax.set_xlim(0.001, analysis_type_to_xmax[self.analysis_type])
            ax.set_xlabel(analysis_type_to_xlabel[self.analysis_type])
        if wtag:
            ax.set_title(tag, loc='right')

        ax.set_ylim(0.)
        ax.set_ylabel('Number of events')

    def _plot_2d(self, ax, wtag=False, **kwargs):
        mesh = ax.pcolormesh(self.bin_edges[0], self.bin_edges[1], self.z.transpose(), zorder=0, cmap=kwargs.pop('cmap', rev_afmhot),  **kwargs)
        cbar = plt.colorbar(mesh, ax=ax)    
        if self.title is not None:
            ax.set_title(sample_to_title[self.title], loc='left', fontsize=20)
        if self.analysis_type is not None:
            _=ax.set_xticks(analysis_type_to_xtickspos[self.analysis_type])
            ax.set_xlim(0.001, analysis_type_to_xmax[self.analysis_type])
            ax.set_xlabel(analysis_type_to_xlabel[self.analysis_type])

        _=ax.set_yticks([30*i for i in range(7)])                          
        ax.set_ylim(0, 180)
        ax.set_ylabel("Angle [degrees]")
        if wtag:
            ax.set_title(tag, loc='right', fontsize=20)
    
    def _rebin_1d(self, new_bin_edges):
        if len(new_bin_edges) != 1:
            raise ValueError(f"The dimension of new edges ({len(new_bin_edges)}) should be equal to 1 as it is 1D sample")
            
        new_xedges = new_bin_edges[0]
        old_xedges = self.bin_edges[0]
        
        if all(new_xedge in old_xedges for new_xedge in new_xedges):
            i = 0
            new_z = np.zeros(new_xedges.shape[0]-1)
            for k in range(self.z.shape[0]):
                if old_xedges[k] >= new_xedges[i+1]:
                    i += 1
                new_z[i] += self.z[k]
            return Sample([new_xedges], new_z, self.title, self.analysis_type)
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
                return Sample( [new_xedges, new_yedges], new_z, self.title, self.analysis_type)
            if len(new_yedges) == 2:
                if self.analysis_type=='e-theta':
                    analysis_type = 'Erec'
                elif self.analysis_type=='PTheta':
                    analysis_type = 'P'
                else:
                    analysis_type = None
                return Sample( [new_xedges], new_z.transpose()[0], self.title, analysis_type)
            if len(new_xedges) == 2:
                if self.analysis_type=='e-theta' or self.analysis_type=='PTheta':
                    analysis_type = 'Theta'
                else:
                    analysis_type = None
                return Sample( [new_yedges], new_z[0], self.title, analysis_type)       
        else:
            raise ValueError("New edges should be contained fully in old edges")

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

    def get_titles(self):
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

    def append(self, sample):
        """
        Adds a new sample to the collection.

        Parameters
        ----------
        sample : Sample
            The `Sample` object to add to the collection.
        """
        self.samples.append(sample)

    def plot(self, ax, sample, wtag=False, **kwargs):
        self.get_sample(sample).plot(ax, wtag=False, **kwargs)


            
            

