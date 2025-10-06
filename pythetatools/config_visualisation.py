import numpy as np
import seaborn as sns

RED = "\033[31m"
RESET = "\033[0m"
GREEN = "\033[32m"

darkblue = np.array([0,102,255])/255
midblue = np.array([51,153,255])/255
lightblue= np.array([153,204,255])/255 
verylightblue= '#c4e2f6'
vermilion = np.array([217,96,59])/255
lightvermilion = np.array([232,163,144])/255
darkvermilion = np.array([0.425, 0.188, 0.115])

midorange = np.array([255,153,51])/255
bluish_green = np.array([0,158,115])/255
light_bluish_green = np.array([0.5, 0.79, 0.575])
dark_bluish_green = np.array([0.0, 0.316, 0.225])
darkorange = np.array([255,102,0])/255
midorange = np.array([255,153,51])/255
lightorange = np.array([255,204,153])/255
verylightorange = np.array([255, 221, 187]) / 255 

rev_afmhot = sns.color_palette("afmhot", as_cmap=True)
rev_afmhot = rev_afmhot.reversed()

t2k_style = {
    'figure.figsize': (9, 6),
    'lines.linewidth': 3,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'axes.axisbelow': True,
    'axes.spines.right': True,
    'xtick.direction': 'in',
    'xtick.labelsize': 25,
    'xtick.top': True,
    'xtick.major.width': 0.8,
    'xtick.major.size': 10,
    'ytick.direction': 'in',
    'ytick.labelsize': 25,
    'ytick.right': True,
    'ytick.major.width': 0.8,
    'ytick.major.size': 10,
    'legend.fancybox': False,
    'legend.fontsize': 20,
    'legend.edgecolor': 'white',
    'legend.shadow': False,
    'grid.linewidth': 0.0,
    'grid.linestyle': '-',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
}

color_mo = {0: midblue, 1: midorange}
mo_to_label = {0: 'Normal ordering', 1: 'Inverted ordering'}
mo_to_title = {0: 'NO', 1: 'IO'}

level_to_hatch = {0.6827:'/', 0.9:'\\\\', 0.9545:'XX', 0.9973:'++++'}
level_to_color = {0:{0.6827:darkblue, 0.9:midblue, 0.9545:lightblue, 0.9973:verylightblue},
                  1:{0.6827:darkorange, 0.9:midorange, 0.9545:lightorange, 0.9973:verylightorange},}
level_to_label = {0.6827:'68.27% C.L.', 0.9:'90.00% C.L.', 0.9545:'95.45% C.L.', 0.9973:'99.73% C.L.'}
level_to_ls = {0.6827:'--', 0.9:'-', 0.9545:'dashdot', 0.9973:'dotted'}

critval_level_to_color = {0: {0.6827: lightblue, 0.9: midblue, 0.9545: darkblue, 0.9973: 'darkblue'}, 
          1: {0.6827: lightorange, 0.9: midorange, 0.9545: darkorange, 0.9973: 'brown'}}

dfbf_to_color = {'bestfit':{0: 'darkblue', 1:'brown' },
                 'datafit':color_mo}

mo_to_colors = {0: [lightblue, midblue,  darkblue,  'darkblue'], 
                1: [lightorange,  midorange, darkorange,  'brown']}

error_band_color = [darkblue, "#E73121"]
error_band_hatch = [ "\\\\",  '///']
flavour_to_xmax_errorbands = {'nue':1.3, 'numu':2.99}
