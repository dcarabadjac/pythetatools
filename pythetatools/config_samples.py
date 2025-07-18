import numpy as np

#Nominal binning edges
erec_egdes = [0.05*i for i in range(61)] + [3.25, 3.50, 3.75, 4.00, 4.50, 5.00, 5.50, 6.00, 7.00, 8.00, 9.00, 10.00, 30.00]
p_edges =  np.arange(0, 1600, 100)
theta_numu_edges = [20*i for i in range(6)] + [180]
theta_nue_edges = [10*i for i in range(15)] + [180]

analysis_type_to_dim = {'Erec':'1D',       'P':'1D',    'e-theta':'2D',             'p-theta':'2D',    'PTheta':'2D'}
analysis_type_xedges = {'Erec':erec_egdes, 'P':p_edges, 'e-theta':erec_egdes,       'p-theta':p_edges, 'PTheta':p_edges}
analysis_type_yedges = {'Erec':None,       'P':None,    'e-theta':theta_numu_edges, 'p-theta':theta_nue_edges, 'PTheta':theta_nue_edges}

#Names used for Sample visualisation
sample_to_nuflav = {'numu1R':'numu', 'nue1R':'nue', 'numubar1R':'numu', 'nuebar1R':'nue','numucc1pi':'numu', 'nuecc1pi':'nue',
                    'nue1RD':'nue', 'numu1R_320kA':'numu', 'nue1R_320kA':'nue', 'numubar1R_320kA':'numu', 'nuebar1R_320kA':'nue', 
                    'numucc1pi':'numu', 'nuecc1pi_320kA':'nue', 'nue1RD_320kA':'nue'}

sample_to_title = {'numu1R':r'FHC 1R$\mu$', 'nue1R':r'FHC 1Re', 'numubar1R':r'RHC 1R$\mu$', 'nuebar1R':r'RHC 1Re',
                   'numucc1pi':r'FHC MR$\mu$', 'nuecc1pi':r'FHC MRe', 'nue1RD':r'FHC 1Re 1d.e.',
                   'numu1R_320kA':r'$\nu_\mu$ 1R 320kA', 'nue1R_320kA':r'$\nu_e$ 1R 320 kA', 'numubar1R_320kA':r'$\bar{\nu}_\mu$ 1R 320 kA',
                   'nuebar1R_320kA':r'$\bar{\nu}_e$ 1R 320 kA', 'numucc1pi_320kA':r'MR $\nu_\mu$CC$1\pi^{+} 320 kA$',
                   'nuecc1pi_320kA':r'$\nu_e$CC$1\pi^{+}$', 'nue1RD_320kA':r'$\nu_e$ 1R 1 d.e. 320kA'}

analysis_type_to_xlabel = {'Erec':r'$E^\nu_{rec}$, [GeV]', 'P':r'$p_e$, [MeV/c]', 'Theta':r'$\theta$, [deg]', 'e-theta':r'$E^\nu_{rec}$, [GeV]', 'p-theta': r'$p_e$, [MeV/c]', 'PTheta': r'$p_e$, [MeV/c]'}
analysis_type_to_xtickspos = {'Erec':np.arange(0, 3.5, 0.5), 'e-theta':np.arange(0, 3.5, 0.5), 'p-theta': np.arange(0, 1300, 200),
                              'P':np.arange(0, 1300, 200), 'Theta':np.arange(0, 200, 20), 'PTheta': np.arange(0, 1300, 200)}
analysis_type_to_xmax = {'Erec':3, 'P':1300 ,'Theta':180, 'e-theta':3, 'p-theta': 1300,  'PTheta':1300}
analysis_type_to_xvar = {'Erec':'Erec', 'P':'p', 'Theta':r'$\theta$', 'e-theta':'Erec', 'p-theta': 'p', 'PTheta': 'p' }
analysis_type_to_energyvar = {'Erec':'Erec', 'P':'p', 'Theta': None, 'e-theta':'Erec', 'p-theta': 'p', 'PTheta': 'p'} 
analysis_type_to_anglevar = {'Erec':None, 'P':None, 'Theta': 'Theta', 'e-theta':'Theta', 'p-theta': 'Theta', 'PTheta':'Theta'}

#Names used for Sample visualisation (breakdown by osc. channel (flavour) and interaction mode)
interaction_modes = ['CC_QE', 'CC_MEC', 'CC_1PIC', 'CC_1PI0', 'CC_COH', 'CC_DIS', 'CC_MPI', 'CC_MISC', 'NC_1PI0', 'NC_1PIC', 'NC_COH', 'NC_GAM', 'NC_OTHER']
osc_channels = ['numu', 'numubar', 'nue_sig', 'nue', 'nuebar', 'nuebar_sig']
interaction_modes_2 = ['CC_QE', 'CC_MEC', 'CC_1PI', 'CC_OTHER', 'NC'] #after int. modes merging
inter2_to_label = {'CC_QE':'CCQE', 'CC_MEC':'CC 2p2h', 'CC_1PI':r'CC 1$\pi$', 
                  'CC_OTHER': 'CC Other', 'NC':'NC'}
inter_to_inter2 = {'CC_QE':'CC_QE', 'CC_MEC':'CC_MEC', 'CC_1PIC':'CC_1PI', 'CC_COH':'CC_1PI',
                'CC_1PI0':'CC_1PI', 'CC_DIS':'CC_OTHER', 'CC_MPI':'CC_OTHER',
                'CC_MISC':'CC_OTHER', 'NC_1PI0': 'NC', 'NC_1PIC':'NC', 'NC_COH':'NC',
                'NC_GAM':'NC', 'NC_OTHER':'NC'}
flavour_to_label = {'numu':r'$\nu_\mu \rightarrow \nu_\mu$', 'numubar':r'$\bar{\nu}_\mu \rightarrow \bar{\nu}_\mu$',
                    'nue': r'$\nu_e \rightarrow \nu_e$', 'nuebar': r'$\bar{\nu}_e \rightarrow \bar{\nu}_e$',
                    'nue_sig': r'$\nu_\mu \rightarrow \nu_e$', 'nuebar_sig': r'$\bar{\nu}_\mu \rightarrow \bar{\nu}_e$'}