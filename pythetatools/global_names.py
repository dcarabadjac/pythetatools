import numpy as np
import os

my_login = 'dcarabad'
my_domain = 'cca.in2p3.fr'

tag = "OARun11A"
library_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir = os.path.join(os.path.dirname(library_dir), 'inputs')
outputs_dir = os.path.join(os.path.dirname(library_dir), 'outputs')

erec_egdes = [0.05*i for i in range(61)] + [3.25, 3.50, 3.75, 4.00, 4.50, 5.00, 5.50, 6.00, 7.00, 8.00, 9.00, 10.00, 30.00]
p_edges =  np.arange(0, 1600, 100)
theta_numu_edges = [20*i for i in range(6)] + [180]
theta_nue_edges = [10*i for i in range(15)] + [180]

analysis_type_to_dim = {'Erec':'1D',       'e-theta':'2D',             'p-theta':'2D'}
analysis_type_xedges = {'Erec':erec_egdes, 'e-theta':erec_egdes,       'p-theta':p_edges }
analysis_type_yedges = {'Erec':None,       'e-theta':theta_numu_edges, 'p-theta':theta_nue_edges }

sample_to_nuflav = {'numu1R':'numu', 'nue1R':'nue', 'numubar1R':'numu', 'nuebar1R':'nue', 'numucc1pi':'numu', 'nuecc1pi':'nue', 'nue1RD':'nue'}
sample_to_title = {'numu1R':r'$\nu_\mu$ 1R', 'nue1R':r'$\nu_e$ 1R', 'numubar1R':r'$\bar{\nu}_\mu$ 1R', 'nuebar1R':r'$\bar{\nu}_e$ 1R', 'numucc1pi':r'MR $\nu_\mu$CC$1\pi^{+}$', 'nuecc1pi':r'$\nu_e$CC$1\pi^{+}$', 'nue1RD':r'$\nu_e$ 1R 1 d.e.'}

analysis_type_to_xlabel = {'Erec':r'$E^\nu_{rec}$, [GeV]', 'e-theta':r'$E^\nu_{rec}$, [GeV]', 'p-theta': r'$p_e$, [MeV]', 'P':r'$p_e$, [MeV]', 'Theta':r'$\theta$, [deg]'}
analysis_type_to_xtickspos = {'Erec':np.arange(0, 3.5, 0.5), 'e-theta':np.arange(0, 3.5, 0.5), 'p-theta': np.arange(0, 1300, 200), 'P':np.arange(0, 1300, 200), 'Theta':np.arange(0, 200, 20)}
analysis_type_to_xmax = {'Erec':3, 'e-theta':3, 'p-theta': 1300, 'P':1300 ,'Theta':180}

analysis_type_to_xvar = {'Erec':'Erec', 'e-theta':'Erec', 'p-theta': 'p', 'P':'p', 'Theta':r'$\theta$'}

analysis_type_to_energyvar = {'Erec':'Erec', 'e-theta':'Erec', 'p-theta': 'p', 'P':'p', 'Theta': None}
analysis_type_to_anglevar = {'Erec':None, 'e-theta':'Theta', 'p-theta': 'Theta', 'P':None, 'Theta': 'Theta'}



mo_to_label = {0: 'Normal Ordering', 1: 'Inverted Ordering'}
osc_param_name = ["delta", "dm2", "sin223", "sin213"]
osc_param_name_to_xlabel = {"delta": {'both': r"$\delta_{CP}$", 0:r"$\delta_{CP}$", 1:r"$\delta_{CP}$"},
                            "dm2":   {'both': r"$\Delta m^2_{32}/|\Delta m^2_{31}|$", 0: r"$\Delta m^2_{32}$", 1:r"$|\Delta m^2_{31}|$"},
                            "sin223":{'both': r"$\sin^{2} \theta_{23}$", 0:r"$\sin^{2} \theta_{23}$", 1:r"$\sin^{2} \theta_{23}$"},
                            "sin213":{'both': r"$\sin^{2} \theta_{13}$", 0:r"$\sin^{2} \theta_{13}$", 1:r"$\sin^{2} \theta_{13}$"}}

osc_param_title = {"delta":[r"$\delta_{CP}$", r"$\delta_{CP}$"], "dm2":[r"$\Delta m^2_{32}$", r"$|\Delta m^2_{31}|$"], "sin223":[r"$\sin^{2} \theta_{23}$", r"$\sin^{2} \theta_{23}$"], "sin213":[r"$\sin^{2} \theta_{13}$", r"$\sin^{2} \theta_{13}$"]}
osc_param_unit = {"delta":"", "dm2": ", $[eV^2/c^4]$", "sin223":"", "sin213":""}

interaction_suffixes = ['CC_QE', 'CC_MEC', 'CC_1PIC', 'CC_1PI0', 'CC_COH', 'CC_DIS', 'CC_MPI', 'CC_MISC', 'NC_1PI0', 'NC_1PIC', 'NC_COH', 'NC_GAM', 'NC_OTHER']

flavours_suffixes = ['numu', 'numubar', 'nue_sig', 'nue', 'nuebar', 'nuebar_sig']

interaction_suffixes_2 = ['CC_QE', 'CC_MEC', 'CC_1PI', 'CC_OTHER', 'NC']

int2_to_label = {'CC_QE':'CCQE', 'CC_MEC':'CC 2p2h', 'CC_1PI':r'CC 1$\pi$', 
                  'CC_OTHER': 'CC Other', 'NC':'NC'}

int1_to_int2 = {'CC_QE':'CC_QE', 'CC_MEC':'CC_MEC', 'CC_1PIC':'CC_1PI', 'CC_COH':'CC_1PI',
                'CC_1PI0':'CC_1PI', 'CC_DIS':'CC_OTHER', 'CC_MPI':'CC_OTHER',
                'CC_MISC':'CC_OTHER', 'NC_1PI0': 'NC', 'NC_1PIC':'NC', 'NC_COH':'NC',
                'NC_GAM':'NC', 'NC_OTHER':'NC'}

flavour_to_label = {'numu':r'$\nu_\mu \rightarrow \nu_\mu$', 'numubar':r'$\bar{\nu}_\mu \rightarrow \bar{\nu}_\mu$',
                    'nue': r'$\nu_e \rightarrow \nu_e$', 'nuebar': r'$\bar{\nu}_e \rightarrow \bar{\nu}_e$',
                    'nue_sig': r'$\nu_\mu \rightarrow \nu_e$', 'nuebar_sig': r'$\bar{\nu}_\mu \rightarrow \bar{\nu}_e$'}



