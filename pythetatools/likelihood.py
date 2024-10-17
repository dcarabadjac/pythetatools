from array import array
import ROOT
import numpy as np
from .global_names import *
from .general_functions import show_minor_ticks
import sys


def read_margtemplates_1D(filename, mo='both'):
    # Open the ROOT file
    file = ROOT.TFile(filename, "READ")
    if file.IsZombie():
        print(f"Error opening file {filename}")
        return None

    # Get the "MargTemplate" tree
    input_tree = file.Get("MargTemplate")
    if not input_tree:
        print(f"'MargTemplate' tree not found in file {filename}")
        return None

    # Determine the number of entries (ngrid)
    nEntries = input_tree.GetEntries()
    print(f"Number of entries in 'MargTemplate': {nEntries}")

    # Prepare a dictionary to hold the branch data
    branches = {}
    branch_types = {
        'Double_t': 'd',  # Double
        'Float_t': 'f',  # Float
        'Int_t': 'i',  # Integer
        'Long_t': 'l'   # Long
    }
    noscparams = 0
    ndiscrparams = 0
    for branch in input_tree.GetListOfBranches():
        branch_name = branch.GetName()
        if branch_name in osc_param_name:
            noscparams += 1
            print("Grid for oscillation parameter found: {}".format(branch_name))
            param_name = branch_name
            branch_type_code = branch.GetLeaf(branch_name).GetTypeName()
        elif branch_name=='mh':
            ndiscrparams = 1
            print("Grid for mh found")          
        else:
            continue
        
        if branch_type_code in branch_types:
            branch_type = branch_types[branch_type_code]
        else:
            print(f"Unsupported branch type: {branch_type_code} for branch {branch_name}")
            continue

    if ndiscrparams==1:
        mo_value = array('d', [0])
        input_tree.SetBranchAddress('mh', mo_value)
    else:
        mo_value = mo
        
    grid_branch_cont = array(branch_type, [0])   
    input_tree.SetBranchAddress(param_name, grid_branch_cont)
    AvNLLtot_branch_cont = array('d', [0])
    input_tree.SetBranchAddress("AvNLLtot", AvNLLtot_branch_cont)
    
    if noscparams > 1:
        raise ValueError("Error: Number of continous osc. params = {} is greater that one, however it is function 1D plot ".format(noscparams))
    elif noscparams ==0:
        raise ValueError("Error: Continous osc. param. not found in the tree  ")
        
    ngrid = nEntries//(1+ndiscrparams)
    print("Number of grid point for {} = {}".format(param_name, ngrid))

    AvNLLtot = {0: np.zeros(ngrid), 1: np.zeros(ngrid)} if ndiscrparams==1 else {mo: np.zeros(ngrid)}
    grid = np.zeros(ngrid)

    for entry in range(nEntries):
        input_tree.GetEntry(entry)
        AvNLLtot[int(mo_value[0])][entry//(ndiscrparams+1)] = AvNLLtot_branch_cont[0]
        grid[entry//(ndiscrparams+1)] = grid_branch_cont[0]

    return grid, AvNLLtot, param_name

class Dchi2:
    def __init__(self, grid, avnllh, param_name, kind='2D'):
        self.grid = grid
        self.avnllh = avnllh
        self.param_name = param_name
        if kind == '2D':
            minimum = {key: min(np.min(array) for array in self.avnllh.values()) for key in self.avnllh.keys()} 
        elif kind == 'conditional':
            minimum = {key: np.min(self.avnllh[key]) for key in self.avnllh.keys()} 
            
        self.dchi2 = {key: 2*(value-minimum[key]) for key, value in self.avnllh.items()}
        if len(list(self.dchi2.keys())) == 2:
            self.mo = 'both'
        else:
            self.mo = self.dchi2.keys()[0]

    def plot(self, ax, mo=None, **kwargs):
        if not mo is None:
            ax.plot(self.grid, self.dchi2[mo], color=color_mo[mo], label=mo_to_label[mo], **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.param_name][mo])
        else:
            for key, value in self.dchi2.items():
                ax.plot(self.grid, value, color=color_mo[key], label=mo_to_label[key], **kwargs)
            ax.set_xlabel(osc_param_name_to_xlabel[self.param_name][self.mo])    
            
        ax.set_ylabel(r'$\Delta \chi^2$')
        show_minor_ticks(ax)
        ax.set_ylim(0)
        ax.legend(edgecolor='white')

    def find_CI(self, nsigma):
        edges_left = []   
        edges_right = []
        c = np.sqrt(nsigma)

        #Treat the case if margin points in inside C.I.
        if self.dchi2[0] <= c:
            edges_left.append(self.grid[0])
        if self.dchi2[-1] <= c:
            edges_right.append(self.grid[-1])

        #Find all the margins of C.I.
        for i in range(len(self.grid) - 1):
            y0, y1 = self.dchi2[i], self.dchi2[i + 1]
            if (y0 - c)>=0 and (y1 - c)<=0:
                x0, x1 = self.grid[i], self.grid[i + 1]
                edge_left = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_left.append(edge_left) 
            if (y0 - c)<=0 and (y1 - c)>=0:
                x0, x1 = self.grid[i], self.grid[i + 1]
                edge_right = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
                edges_right.append(edge_right) 
                
        return edges_left, edges_right

