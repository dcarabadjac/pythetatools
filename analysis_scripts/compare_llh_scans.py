import pythetatools.toyxp as toyxp
import pythetatools.likelihood as likelihood
import pythetatools.file_manager as file_manager
import uproot



def load_sigmavarskdet_per_sample(filepath, dict_df_names, group, filetype, kind='sample'):
    #This function was only tested for SK det group. Minor modification would be necessary to generalise for the each group.
    
    llh_scan_toy = toyxp.ToyXp() #Even though it is llh value we use ToyXp to handle the histogram object (not continous line)
    if filetype=='P-theta':
        with uproot.open(filepath) as file:   
            for parameter in dict_df_names[group]['P-theta'].values:
                if kind == 'sample':
                    histname = f"hscan_{parameter}"             
                else:
                    histname = f"hscan_{parameter}_constraint"             
                if histname in file:  
                    hist = file[histname]
                    xedges = hist.axis(0).edges()
                    z = hist.values()
                    llh_scan_toy.append(toyxp.Sample([xedges], z, title=histname, sample_title='nue1R'))
                else:
                    print(f'{histname} NOT FOUND')
                    
    elif filetype=='MaCh-3':
        with uproot.open(filepath) as file:   
            for iparam, parameter in enumerate(dict_df_names[group]['P-theta'].values):
                if kind == 'sample':
                    directory = file["Sample_LLH"]
                    histname = f"skd_joint_{iparam}_sam"
                else:
                    directory = file["skdet_LLH"]
                    histname = f"skd_joint_{iparam}_skdet"
                if histname in directory:  
                    hist = directory[histname]
                    xedges = hist.axis(0).edges()
                    z = hist.values()
                    llh_scan_toy.append(toyxp.Sample([xedges], z, title=histname, sample_title='nue1R'))
                else:
                    print(f'{histname} NOT FOUND')
    return llh_scan_toy