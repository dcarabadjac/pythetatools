import numpy as np
import os

my_login = 'dcarabad'
my_domain = 'cca.in2p3.fr'

tag = "T2K preliminary 2023"
library_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir = os.path.join(os.path.dirname(library_dir), 'inputs')
outputs_dir = os.path.join(os.path.dirname(library_dir), 'outputs')

erec_egdes = [0.05*i for i in range(61)] + [3.25, 3.50, 3.75, 4.00, 4.50, 5.00, 5.50, 6.00, 7.00, 8.00, 9.00, 10.00, 30.00]
p_edges =  np.arange(0, 1300, 100)
theta_numu_edges = [20*i for i in range(6)] + [180]
theta_nue_edges = [10*i for i in range(15)] + [180]

analysis_type_to_dim = {'Erec':'1D',       'e-theta':'2D',             'PTheta':'2D'}
analysis_type_xedges = {'Erec':erec_egdes, 'e-theta':erec_egdes,       'PTheta':p_edges }
analysis_type_yedges = {'Erec':None,       'e-theta':theta_numu_edges, 'PTheta':theta_nue_edges }

sample_to_nuflav = {'numu1R':'numu', 'nue1R':'nue', 'numubar1R':'numu', 'nuebar1R':'nue', 'numucc1pi':'numu', 'nuecc1pi':'nue', 'nue1RD':'nue'}
sample_to_title = {'numu1R':r'$\nu_\mu$ 1R', 'nue1R':r'$\nu_e$ 1R', 'numubar1R':r'$\bar{\nu}_\mu$ 1R', 'nuebar1R':r'$\bar{\nu}_e$ 1R', 'numucc1pi':r'MR $\nu_\mu$CC$1\pi^{+}$', 'nuecc1pi':r'1R $\nu_e$CC$1\pi^{+}$', 'nue1RD':r'1R $\nu_e$ 1 d.e.$'}

analysis_type_to_xlabel = {'Erec':'Energy [GeV]', 'e-theta':'Energy [GeV]', 'PTheta': 'Electron momentum [MeV]', 'P':'Electron momentum [MeV]'}
analysis_type_to_xtickspos = {'Erec':np.arange(0, 3.5, 0.5), 'e-theta':np.arange(0, 3.5, 0.5), 'PTheta': np.arange(0, 1300, 200), 'P':np.arange(0, 1300, 200)}
analysis_type_to_xmax = {'Erec':3, 'e-theta':3, 'PTheta': 1300, 'P':1300}

analysis_type_to_xvar = {'Erec':'Erec', 'e-theta':'Erec', 'PTheta': 'p', 'P':'p'}

mo_to_label = {0: 'Normal Ordering', 1: 'Inverted Ordering'}
osc_param_name = ["delta", "dm2", "sin223", "sin213"]
osc_param_name_to_xlabel = {"delta": {'both': r"$\delta_{CP}$", 0:r"$\delta_{CP}$", 1:r"$\delta_{CP}$"},
                            "dm2":   {'both': r"$\Delta m^2_{32}/|\Delta m^2_{31}|$", 0: r"$\Delta m^2_{32}$", 1:r"$|\Delta m^2_{31}|$"},
                            "sin223":{'both': r"$\sin^{2} \theta_{23}$", 0:r"$\sin^{2} \theta_{23}$", 1:r"$\sin^{2} \theta_{23}$"},
                            "sin213":{'both': r"$\sin^{2} \theta_{13}$", 0:r"$\sin^{2} \theta_{13}$", 1:r"$\sin^{2} \theta_{13}$"}}

osc_param_title = {"delta":[r"$\delta_{CP}$", r"$\delta_{CP}$"], "dm2":[r"$\Delta m^2_{32}$", r"$|\Delta m^2_{31}|$"], "sin223":[r"$\sin^{2} \theta_{23}$", r"$\sin^{2} \theta_{23}$"], "sin213":[r"$\sin^{2} \theta_{13}$", r"$\sin^{2} \theta_{13}$"]}
osc_param_unit = {"delta":"", "dm2": ", $[eV^2/c^4]$", "sin223":"", "sin213":""}





