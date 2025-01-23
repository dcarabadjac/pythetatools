from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import subprocess
import os
import uproot
import subprocess

def check_remote_path(remote_path, login, domain):
    # Check if the remote path is a file or directory using SSH
    check_file_command = f"ssh {login}@{domain} 'test -f \"{remote_path}\" && echo file || echo notfile'"
    check_dir_command = f"ssh {login}@{domain} 'test -d \"{remote_path}\" && echo dir || echo notdir'"

    # Run file check
    file_check = subprocess.run(check_file_command, shell=True, capture_output=True, text=True)
    if file_check.returncode == 0 and file_check.stdout.strip() == 'file':
        return 'file'

    # Run directory check
    dir_check = subprocess.run(check_dir_command, shell=True, capture_output=True, text=True)
    if dir_check.returncode == 0 and dir_check.stdout.strip() == 'dir':
        return 'dir'

    return None  # If neither a file nor directory exists

def create_destination_folder(destination):
    # Ensure the destination folder exists
    folder_path = os.path.dirname(destination) if os.path.isfile(destination) else destination
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created destination folder: {folder_path}")
    else:
        print(f"Destination folder already exists: {folder_path}")


def download(input_path, login, domain, destination, overwrite=False):
    """Dowloads a file or directory from computing center's server
    
    Parameters
    ----------
    input_path : string
        Path on the server to the inputs
    login : string
        Login of the account where the inputs are stored
    domain : string
        Domain of the computing center
    destination : string
        Output directory for the downloaded inputs
    """
    # Check if the remote path is a file or directory
    remote_type = check_remote_path(input_path, login, domain)

    if remote_type is None:
        print(f"Error: {input_path} does not exist or is not a file or directory on the remote system.")
        return

    # Extract the remote name
    basename = os.path.basename(input_path)

    # Local full path for the file or directory to be downloaded
    local_path = os.path.join(destination, basename)

    if not overwrite:
        # Check if a file or directory with the same name already exists in the destination
        if os.path.exists(local_path):
            print(f"{'File' if os.path.isfile(local_path) else 'Directory'} already exists in destination: {local_path}")
            return

    # Create destination folder if it does not exist
    create_destination_folder(destination)

    # Construct SCP command based on the type of remote path
    if remote_type == 'file':
        scp_command = f"scp {login}@{domain}:{input_path} {destination}"
    elif remote_type == 'dir':
        scp_command = f"scp -r {login}@{domain}:{input_path} {destination}"

   # Execute the SCP command
    try:
        subprocess.run(scp_command, shell=True, check=True)
        print(f"Downloaded {input_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during SCP transfer: {e}")


def read_histogram(filename, histname, dim):
    """Loads 1D or 2D histogram from the ROOT file."""
    
    with uproot.open(filename) as file:
        hist = file[f'{histname}']
        xedges = hist.axis(0).edges()
        z = hist.values()
        if dim==2:
            yedges = hist.axis(1).edges()
    if dim==2:  
        return xedges, yedges, z
    if dim==1:  
        return xedges, z


