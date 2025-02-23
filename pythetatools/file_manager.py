from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import subprocess
import os
import uproot
import subprocess
from .global_names import my_login, my_domain

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
        return 'directory'

    return None  # If neither a file nor directory exists


def create_destination_folder(destination):
    # Ensure the destination folder exists
    folder_path = destination
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created destination folder: {folder_path}")
    else:
        print(f"Destination folder already exists: {folder_path}")


def download(input_path, destination, new_name=None, pattern='*', login="my_login", domain="my_domain", overwrite=False):
    """Downloads a file or directory from the computing center's server with progress indication.
    
    Parameters
    ----------
    input_path : string
        Path on the server to the input file or directory.
    destination : string
        Output directory for the downloaded inputs.
    new_name : string, optional
        New name for the downloaded file (only applicable for files).
    pattern : string, optional
        Pattern to match files when downloading a directory (default is '*', meaning all files).
    login : string
        Login of the account where the inputs are stored.
    domain : string
        Domain of the computing center.
    overwrite : bool, optional
        Whether to overwrite existing files, default is False.
    """

    # Check if the remote path is valid
    remote_type = check_remote_path(input_path, login, domain)
    print(remote_type)
    if remote_type == "not_found":
        print(f"Error: {input_path} does not exist on the remote system.")
        return

    # Ensure destination folder exists
    create_destination_folder(destination)

    if remote_type == "directory":
        # If input_path is a directory, sync it to destination with a pattern
        rsync_command = f"rsync -ah --progress --include='{pattern}' --exclude='*' {login}@{domain}:{input_path}/ {destination}/"

    elif remote_type == "file":
        # Determine local file path
        local_file_path = os.path.join(destination, new_name) if new_name else os.path.join(destination, os.path.basename(input_path))

        # Check if file exists and handle overwrite
        if not overwrite and os.path.exists(local_file_path):
            print(f"File already exists: {local_file_path}. Use overwrite=True to replace it.")
            return

        # Rsync for a single file
        rsync_command = f"rsync -ah --progress {login}@{domain}:{input_path} {local_file_path}"

    else:
        print(f"Error: {input_path} is neither a file nor a directory.")
        return

    # Execute rsync command
    print(f"Executing: {rsync_command}")
    subprocess.run(rsync_command, shell=True)



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


