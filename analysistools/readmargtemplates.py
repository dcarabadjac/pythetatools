from array import array
import ROOT
import numpy as np
from .global_names import *
import subprocess

#Downloads already merged (by throws) margtemplate file
def download_merged_margtemplates(dirname, statonly=False):
    if not statonly:
        scp_command = f"scp dcarabad@cca.in2p3.fr:" \
                  f"/sps/t2k/dcarabad/Develop/OA2024/P-theta/inputs/outputs/{dirname}/merged.root Root_files/{dirname}_merged.root"
    else:
        scp_command = f"scp dcarabad@cca.in2p3.fr:" \
                  f"/sps/t2k/dcarabad/Develop/OA2024/P-theta/inputs/outputs/{dirname}/margtemplates_delta_100k_b0000_t000000.root Root_files/{dirname}_merged.root"
    # Execute the SCP command using subprocess
    subprocess.run(scp_command, shell=True)

def read_asimov_fit_1D(filename):
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
            grid_osc_param = branch_name
            branch_type_code = branch.GetLeaf(branch_name).GetTypeName()
        elif branch_name=='mh':
            ndiscrparams = 1
            print("Grid for mh found")
            branch_type_code = branch.GetLeaf(branch_name).GetTypeName()
        else:
            continue
        
        if branch_type_code in branch_types:
            branch_type = branch_types[branch_type_code]
        else:
            print(f"Unsupported branch type: {branch_type_code} for branch {branch_name}")
            continue

        grid_branch_cont = array(branch_type, [0])
        input_tree.SetBranchAddress(grid_osc_param, grid_branch_cont)

    AvNLLtot_branch_cont = array('d', [0])
    input_tree.SetBranchAddress("AvNLLtot", AvNLLtot_branch_cont)
    
    if noscparams > 1:
        raise ValueError("Error: Number of continous osc. params = {} is greater that one, however it is function 1D plot ".format(noscparams))
    elif noscparams ==0:
        raise ValueError("Error: Continous osc. param. not found in the tree  ")
        
    ngrid = nEntries//(1+ndiscrparams)
    print("Number of grid point for {} = {}".format(grid_osc_param, ngrid))

    AvNLLtot = np.zeros((ndiscrparams+1, ngrid))
    osc_array_grid = np.zeros(ngrid)

    for entry in range(nEntries):
        input_tree.GetEntry(entry)
        AvNLLtot[entry%(ndiscrparams+1)][entry//(ndiscrparams+1)] = AvNLLtot_branch_cont[0]
        osc_array_grid[entry//(ndiscrparams+1)] = grid_branch_cont[0]

    return AvNLLtot, osc_array_grid, grid_osc_param
    