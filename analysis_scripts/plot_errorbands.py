from pythetatools import config as cfg
from pythetatools.config_visualisation import *
from pythetatools.config_samples import sample_to_title, sample_to_nuflav
from pythetatools.base_visualisation import *
from pythetatools.file_manager import read_histogram
import pythetatools.toyxp as toyxp
import matplotlib.patches as patches



def plot_errorbands(indirs, labels, outdir_path, group='all', nfiles = 100, save=True):
    sample_titles = cfg.CONFIG.sample_titles
    asimov_dummy = toyxp.load(f"{indirs[0]}/ErrorBands_00.root", kind='asimov', sample_titles=sample_titles)
    
    for sample_title in sample_titles:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        error_bars = []
        handles = []
        for indir in indirs:
            z_mean = np.zeros(len(asimov_dummy[sample_title].z))
            errors_mean = np.zeros(len(asimov_dummy[sample_title].z))
            for i in range(nfiles):
                bin_edges, z, errors = read_histogram(f"{indir}/ErrorBands_{i:02d}.root", f'h{sample_title}_errorbands_{group}')
                z_mean += z
                errors_mean += errors  
            z_mean = z_mean/nfiles
            errors_mean = errors_mean/nfiles
            
            sample_mean  = toyxp.Sample(bin_edges, z_mean, title=sample_title, analysis_type='Erec')
            sample_minus = sample_mean - errors_mean
            sample_plus  = sample_mean + errors_mean
            error_bars.append([sample_minus, sample_plus])

        sample_mean.plot(ax, color='none') # to set the axis labels
       

        for k in range(len(error_bars)):
            sample_m, sample_p = error_bars[k]
            bin_edges = sample_m.bin_edges[0]
            for i in range(len(sample_m.z)):
                x_left = bin_edges[i]
                width = bin_edges[i + 1] - bin_edges[i]

                y_min = min(sample_m.z[i], sample_p.z[i])
                y_max = max(sample_m.z[i], sample_p.z[i])
                height = y_max - y_min

                rect = patches.Rectangle((x_left, y_min), width, height, edgecolor=error_band_color[k],
                                         hatch=error_band_hatch[k], facecolor='none', linewidth=1)
                ax.add_patch(rect)
                ax.scatter(x_left+width/2, y_min+height/2, color=error_band_color[k], marker='s', s=3)

            handles.append(patches.Patch(edgecolor=error_band_color[k],
                                         hatch=error_band_hatch[k], facecolor='none', label=labels[k]))
            ax.set_title(sample_to_title[sample_title], loc='left')
            ax.set_title(cfg.CONFIG.tag, loc='right')
            ax.set_xlim(0.001, flavour_to_xmax_errorbands[sample_to_nuflav[sample_title]])
        
        show_minor_ticks(ax)
        fig.set_size_inches(8, 6)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.5)
        ax.legend(handles=handles, loc='best')

        if save:
            if len(labels) > 1:
                suffix = f"{labels[0]}_vs_{labels[1]}"
            else:
                suffix = f"{labels[0]}"
            fig.savefig(f'{outdir_path}/ErrorBars_{sample_title}_{suffix}.pdf', bbox_inches='tight')
