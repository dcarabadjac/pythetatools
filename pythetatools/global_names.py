import numpy as np
import seaborn as sns

tag = "T2K preliminary 2023"

erec_binning_nominal = np.arange(0, 3.05, 0.05)

sample_to_nuflav = {'numu1R':'numu', 'nue1R':'nue', 'numubar1R':'numu', 'nuebar1R':'nue', 'numucc1pi':'numu', 'nue1RD':'nue'}
sample_to_title = {'numu1R':r'$\nu_\mu$ 1R', 'nue1R':r'$\nu_e$ 1R', 'numubar1R':r'$\bar{\nu}_\mu$ 1R', 'nuebar1R':r'$\bar{\nu}_e$ 1R', 'numucc1pi':r'MR $\nu_\mu$CC$1\pi^{+}$', 'nue1RD':r'1R $\nu_e$CC$1\pi^{+}$'}

binning_to_xlabel = {'Erec':'Energy [GeV]', 'e-theta':'Energy [GeV]', 'p-theta': 'Electron momentum [MeV]'}
binning_to_xtickspos = {'Erec':np.arange(0, 3.5, 0.5), 'e-theta':np.arange(0, 3.5, 0.5), 'nue': np.arange(0, 1300, 200)}
binning_to_xmax = {'Erec':3, 'e-theta':3, 'p-theta': 1300}

binning_to_xvar = {'Erec':'Erec', 'e-theta':'Erec', 'PTheta': 'p'}

mo_to_label = {0: 'Normal Ordering', 1: 'Inverted Ordering'}
osc_param_name = ["delta", "dm2", "sin223", "sin213"]
osc_param_name_to_xlabel = {"delta": {'both': r"$\delta_{CP}$", 0:r"$\delta_{CP}$", 1:r"$\delta_{CP}$"},
                            "dm2":   {'both': r"$\Delta m^2_{32}/|\Delta m^2_{31}|$", 0: r"$\Delta m^2_{32}$", 1:r"$|\Delta m^2_{31}|$"},
                            "sin223":{'both': r"$\sin^{2} \theta_{23}$", 0:r"$\sin^{2} \theta_{23}$", 1:r"$\sin^{2} \theta_{23}$"},
                            "sin213":{'both': r"$\sin^{2} \theta_{13}$", 0:r"$\sin^{2} \theta_{13}$", 1:r"$\sin^{2} \theta_{13}$"}}

osc_param_title = {"delta":[r"$\delta_{CP}$", r"$\delta_{CP}$"], "dm2":[r"$\Delta m^2_{32}$", r"$|\Delta m^2_{31}|$"], "sin223":[r"$\sin^{2} \theta_{23}$", r"$\sin^{2} \theta_{23}$"], "sin213":[r"$\sin^{2} \theta_{13}$", r"$\sin^{2} \theta_{13}$"]}
osc_param_unit = {"delta":"", "dm2": ", $[eV^2/c^4]$", "sin223":"", "sin213":""}

RED = "\033[31m"
RESET = "\033[0m"
GREEN = "\033[32m"

darkblue = np.array([0,102,255])/255
midblue = np.array([51,153,255])/255
lightblue= np.array([153,204,255])/255 
verylightblue= '#c4e2f6'
vermilion = np.array([217,96,59])/255
midorange = np.array([255,153,51])/255
bluish_green = np.array([0,158,115])/255
darkorange = np.array([255,102,0])/255
midorange = np.array([255,153,51])/255
lightorange = np.array([255,204,153])/255

color_mo = {0: midblue, 1: midorange}

rev_afmhot = sns.color_palette("afmhot", as_cmap=True)
rev_afmhot = rev_afmhot.reversed()

t2k_style = {
    'figure.figsize': (9, 6),
    'lines.linewidth': 2,
    'axes.labelsize': 20,
    'axes.titlesize': 25,
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'axes.axisbelow': True,
    'axes.spines.right': True,
    'xtick.direction': 'in',
    'xtick.labelsize': 20,
    'xtick.top': True,
    'xtick.major.width': 0.8,
    'xtick.major.size': 10,
    'ytick.direction': 'in',
    'ytick.labelsize': 20,
    'ytick.right': True,
    'ytick.major.width': 0.8,
    'ytick.major.size': 10,
    'legend.fancybox': False,
    'legend.fontsize': 16,
    'legend.shadow': False,
    'grid.linewidth': 0.0,
    'grid.linestyle': '-',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
}