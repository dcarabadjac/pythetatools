import os

#Only needed if you want to download files from CC
my_login = 'dcarabad'
my_domain = 'cca.in2p3.fr'

# tag used for the plots
tag = "OARun11A preliminary"
#tag = "Hyper-K preliminary"

#Local pathes to the library, inputs and outpits directory
library_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir = os.path.join(os.path.dirname(library_dir), 'inputs')
outputs_dir = os.path.join(os.path.dirname(library_dir), 'outputs')