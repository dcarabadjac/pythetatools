import numpy as np
from matplotlib import pyplot as plt
from .global_names import tag, nuflav_to_xlabel, nuflav_to_xtickspos, nuflav_to_xmax, sample_to_nuflav, sample_to_title, rev_afmhot, nuflav_to_xvar
import subprocess
from array import array
import ROOT

def get_toy(filename):
    toy = ToyXp([])
    file = ROOT.TFile(filename, "READ")
    
    keys = file.GetListOfKeys()
    
    for key in keys:
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TH2D):  
            xedges, yedges, z = get_hist2D(filename, obj.GetName())
            toy.append_sample(Sample2D(obj.GetName()[1:], xedges, yedges, z))
    for key in keys:  
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TH1D) and (not (obj.GetName()[1:] in toy.get_titles())):
            xedges, z = get_hist1D(filename, obj.GetName())
            toy.append_sample(Sample1D(obj.GetName()[1:], xedges, z))
    return toy

def bin_sample1D(title, xvar, xedges):
    hist = np.histogram(xvar, bins=xedges)
    return Sample1D(title, hist[1], hist[0])

def get_dchi2(toy_obs, toy_exp, perbin=False):
    zero_mask = (toy_obs.z == 0)
    result = np.zeros_like(toy_obs.z, dtype=float)
    non_zero_mask = ~zero_mask
    result[non_zero_mask] = toy_exp.z[non_zero_mask]-toy_obs.z[non_zero_mask] - toy_obs.z[non_zero_mask]* \
                                np.log(toy_exp.z[non_zero_mask]/toy_obs.z[non_zero_mask]) 
    result[zero_mask] = toy_exp.z[zero_mask]
    if not perbin:
        result = np.sum(result)
    return result

def download_toyxp(path):

    scp_command = f"scp dcarabad@cca.in2p3.fr:" \
                  f"{path} /Users/denis.carabadjac/Python/analysistools/inputs/ToyXp/"
    subprocess.run(scp_command, shell=True)

def get_tree(filename, sample_title, ientry):

    file = ROOT.TFile(filename, "READ")
    
    if file.IsZombie():
        print(f"Error opening file {filename}")
        return None

    input_tree = file.Get("ToyXp")
    if not input_tree:
        print(f"'ToyXp' tree not found in file {filename}")
        return None
    xvar_branch =  ROOT.std.vector('double')()
    t_branch =  ROOT.std.vector('double')()

    input_tree.SetBranchAddress(f"{nuflav_to_xvar[sample_to_nuflav[sample_title]]}_{sample_title}", xvar_branch)
    input_tree.SetBranchAddress(f"t_{sample_title}", t_branch)

    nEntries = input_tree.GetEntries()
  
    input_tree.GetEntry(ientry)
    xvar = xvar_branch
    theta = t_branch
    return np.array(xvar), np.array(theta)

def get_hist2D(filename, histname):
    # Open the files
    file = ROOT.TFile(filename, "READ")
    
    if not file.IsOpen():
        print("Error opening one or more files.")
        exit(1)
    
    # Read TH2D histograms from the second file
    th2d = file.Get(histname)
    
    if not th2d:
        print("Error reading one or more histograms.")
        exit(2)

    # Convert TH2D content to NumPy arrays
    bin_edges_th2d_x = np.array([th2d.GetXaxis().GetBinLowEdge(i) for i in range(1, th2d.GetNbinsX() + 2)])
    bin_edges_th2d_y = np.array([th2d.GetYaxis().GetBinLowEdge(i) for i in range(1, th2d.GetNbinsY() + 2)])
    hist_content_th2d = np.array([[th2d.GetBinContent(i, j) for j in range(1, th2d.GetNbinsY() + 1) ] for i in range(1, th2d.GetNbinsX() + 1)])

    # Close the files
    file.Close()
    return bin_edges_th2d_x, bin_edges_th2d_y, hist_content_th2d

def get_hist1D(filename, histname):
    # Open the files
    file = ROOT.TFile(filename, "READ")
    
    if not file.IsOpen():
        print("Error opening one or more files.")
        exit(1)
    
    # Read TH2D histograms from the second file
    th1d = file.Get(histname)
    
    if not th1d:
        print("Error reading one or more histograms.")
        exit(2)

    # Convert TH2D content to NumPy arrays
    bin_edges_th1d_x = np.array([th1d.GetXaxis().GetBinLowEdge(i) for i in range(1, th1d.GetNbinsX() + 2)])
    hist_content_th1d = np.array([th1d.GetBinContent(i)  for i in range(1, th1d.GetNbinsX() + 1)])

    # Close the files
    file.Close()
    return bin_edges_th1d_x, hist_content_th1d


class Sample:
    def __init__(self, title, z):
        z = np.array(z)
        self.__z = z
        self.__title = title
        self.__nbins = self.z.size
        self.__shape = self.z.shape
        self.__nuflav = sample_to_nuflav[title]
    def nevents(self):
        return np.sum(self.z)
    def print(self):
        print(f"Sample:{self.title}; shape:{self.shape}; Total number of events:{self.nevents()}")
    @property
    def z(self):
        return self.__z
    @property
    def nbins(self):
        return self.__nbins
    @property
    def shape(self):
        return self.__shape
    @property
    def nuflav(self):
        return self.__nuflav
    @property
    def title(self):
        return self.__title
    def set_z(self, new_z):
        self.__z = new_z

class Sample1D(Sample):
    def __init__(self, title, xedges, z):
        super().__init__(title, z)
        self.__xedges = np.array(xedges)
    @property
    def xedges(self):
        return self.__xedges
    def set_xedges(self, new_xedges):
        self.__xedges = new_xedges
        
    def plot(self, ax, wtag=False, **kwargs):
        xcenters = (self.xedges[1:] + self.xedges[:-1])/2
        widths = (self.xedges[1:] - self.xedges[:-1])
        ax.bar(xcenters, self.z, width=widths, zorder=0, **kwargs)
        ax.set_title(sample_to_title[self.title], loc='left')
        if wtag:
            ax.set_title(tag, loc='right')
        _=ax.set_xticks(nuflav_to_xtickspos[self.nuflav])
        ax.set_xlim(0.001, nuflav_to_xmax[self.nuflav])
        ax.set_xlabel(nuflav_to_xlabel[self.nuflav])
        ax.set_ylabel('Number of events')
        
    def rebin(self, new_xedges):
        if all(new_xedge in self.xedges for new_xedge in new_xedges):
            i = 0
            new_z = np.zeros(new_xedges.shape[0]-1)
            for k in range(self.shape[0]):
                if self.xedges[k] >= new_xedges[i+1]:
                    i += 1
                new_z[i] += self.z[k]
            return Sample1D(self.title, new_xedges, new_z)
        else:
            raise ValueError("New edges should be contained fully in old edges")

    def slice(self, xmin, xmax):
        start = np.where(self.xedges == xmin)[0][0]
        stop = np.where(self.xedges == xmax)[0][0]
        new_xedges = self.xedges[start:stop+1]
        new_z = self.z[start:stop]
        return Sample1D(self.title, new_xedges, new_z)


    
class Sample2D(Sample):
    def __init__(self, title, xedges, yedges, z):
        super().__init__(title, z)
        self.__xedges = xedges
        self.__yedges = yedges
    @property
    def xedges(self):
        return self.__xedges
    def set_xedges(self, new_xedges):
        self.__xedges = new_xedges
    @property
    def yedges(self):
        return self.__yedges
    def set_yedges(self, new_yedges):
        self.__yedges = new_yedges
        
    def plot(self, ax, wtag=False, **kwargs):
        mesh = ax.pcolormesh(self.xedges, self.yedges, self.z.transpose(), zorder=0, cmap=kwargs.pop('cmap', rev_afmhot),  **kwargs)
        cbar = plt.colorbar(mesh, ax=ax)        
        ax.set_title(sample_to_title[self.title], loc='left', fontsize=20)
        if wtag:
            ax.set_title(tag, loc='right', fontsize=20)
        _=ax.set_xticks(nuflav_to_xtickspos[self.nuflav])
        _=ax.set_yticks([30*i for i in range(7)]) #Make ticks step 30 degrees
        ax.set_xlim(0.001, nuflav_to_xmax[self.nuflav])
        ax.set_ylim(0, 180)
        ax.set_xlabel(nuflav_to_xlabel[self.nuflav])
        ax.set_ylabel("Angle [degrees]")
        
    def rebin(self, new_xedges, new_yedges):
        if all(new_xedge in self.xedges for new_xedge in new_xedges) and all(new_yedge in self.yedges for new_yedge in new_yedges):
            new_z = np.zeros((new_xedges.shape[0]-1, new_yedges.shape[0]-1))
            j = 0
            for m in range(self.shape[1]):
                if self.yedges[m] >= new_yedges[j+1]:
                    j+=1
                i = 0
                for k in range(self.shape[0]):
                    if self.xedges[k] >= new_xedges[i+1]:
                        i += 1
                    new_z[i][j] += self.z[k][m]
            if new_yedges.shape[0] > 2 and new_xedges.shape[0] > 2:
                return Sample2D(self.title, new_xedges, new_yedges, new_z)
            if new_xedges.shape[0] > 2:
                return Sample1D(self.title, new_xedges, new_z.transpose()[0])
            if new_yedges.shape[0] > 2:
                return Sample1D(self.title, new_yedges, new_z[0])
            
        else:
            raise ValueError("New edges should be contained fully in old edges")

class ToyXp:
    def __init__(self, samples_list):
        self.samples = samples_list

    def get_titles(self):
        titles = []
        for sample in self.samples:
            titles.append(sample.title)
        return titles
    
    def get_sample(self, title):
        for sample in self.samples:
            if sample.title == title:
                return sample
        raise ValueError("Sample {title} not found in this toy")

    def print(self):
        print(f'{len(self.samples)} samples are included in this toy:')
        for sample in self.samples:
            sample.print()

    def append_sample(self, sample):
        self.samples.append(sample)

            
            
