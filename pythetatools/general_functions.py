from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.transforms as transforms
import ROOT
import subprocess


def download_file(inputpath, outputdir, login, host='cca.in2p3.fr'):
    r"""Downloads a file from a remote SSH

    Parameters
    ----------
    inputpath: the path on the remote SSH to the input file
    outputdir: output directory for downloaded file
    login: user login on the remote SSH where the file is stored
    host: name of SSH
    
    """
    scp_command = f"scp {login}@{host}:/{inputpath} {outputdir}"
    # Execute the SCP command using subprocess
    subprocess.run(scp_command, shell=True)


def find_roots(x, y, c):
    """
    Find the roots of the equation y = c using linear interpolation between points.
    
    Parameters:
    x (list or numpy array of float): List or array of x-coordinates.
    y (list or numpy array of float): List or array of y-coordinates.
    c (float): The y-value for which to find the x-coordinates where y = c.
    
    Returns:
    list of float: The x-coordinates where y = c.
    """
    # Ensure the input types are consistent and convert lists to numpy arrays if needed
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    
    # Ensure the lists/arrays are non-empty and 1D, and of the same length
    if x.size == 0 or y.size == 0:
        raise ValueError("Input lists/arrays must be non-empty.")
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError("Input lists/arrays should be 1D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input lists/arrays must be of the same length.")
    
    roots = []
    
    # Iterate through pairs of points
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        
        # Check if the function crosses y = c between y0 and y1
        if (y0 - c) * (y1 - c) < 0:
            x0, x1 = x[i], x[i + 1]
            root = x0 + (x1 - x0) * (c - y0) / (y1 - y0)
            roots.append(root)    
    return roots
    
def get_1sigma_interval(x_data, y_data):
    roots = find_roots(x_data, y_data)
    if len(roots)==2: 
        return abs(roots[1] - roots[0])/2
    elif len(roots)==0:
        return None
    elif len(roots)==4:
        print(f"4 roots found. Check if they look adequate:{roots}")
        return abs(roots[3] - roots[2])/2    
    else:
        print("Bad roots founds")

def show_minor_ticks(ax, axis='both'):
    """
    Enable minor ticks on the specified axis of a plot.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to enable minor ticks.
    axis : str, optional
        Specifies which axis to enable minor ticks on: 'x', 'y', or 'both' (default is 'both').

    """
    # Define a function to set minor ticks for a specific axis
    def set_minor_ticks(axis_obj):
        axis_obj.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=6, width=0.8)

    # Enable minor ticks based on the specified axis
    if axis in ['both', 'x']:
        set_minor_ticks(ax.xaxis)
    if axis in ['both', 'y']:
        set_minor_ticks(ax.yaxis)
    if axis not in ['both', 'x', 'y']:
        print("Unknown axis: expected 'x', 'y', or 'both'")


def shift_bbox_center(ax, dx, dy, exp):
    """
    Shift the center of an axis bounding box by a specified amount and expand it.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes object whose bounding box is to be shifted.
    dx : float
        The amount to shift the bounding box center in the x-direction.
    dy : float
        The amount to shift the bounding box center in the y-direction.
    exp : tuple of floats
        A tuple containing two values by which to expand the bounding box 
        (scale factors for the width and height).

    Returns:
    -------
    matplotlib.transforms.Bbox
        The transformed bounding box, shifted and expanded as specified.
    
    """
    # Get the current bounding box in display coordinates and invert it to data coordinates
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    # Calculate current center of the bounding box
    current_center = bbox.get_points().mean(axis=0)

    # Calculate new center position
    new_center = current_center + [dx, dy]

    # Calculate the amount of shift in x and y directions
    shift_x = new_center[0] - current_center[0]
    shift_y = new_center[1] - current_center[1]

    # Create a transform object for shifting the bounding box
    translate = transforms.Affine2D().translate(shift_x, shift_y)

    # Apply the translation transform to the bounding box
    bbox_transformed = bbox.transformed(translate)

    # Return the expanded bounding box
    return bbox_transformed.expanded(*exp)

def read_histogram(filename, histname):
    # Open the files
    file = ROOT.TFile(filename, "READ")
    
    if not file.IsOpen():
        print("Error opening one or more files.")
        exit(1)
    
    # Read TH2D histograms from the second file
    th1d = file.Get(histname)
    
    if not th1d:
        print("Error reading one or more histograms.")
        exit(2)

    bin_edges_th1d_x = np.array([th1d.GetXaxis().GetBinCenter(i) for i in range(1, th1d.GetNbinsX() + 1)])
    hist_content_th1d = np.array([[th1d.GetBinContent(i, j) for j in range(1, th1d.GetNbinsY() + 1) ] for i in range(1, th1d.GetNbinsX() + 1)])

    # Close the files
    file.Close()
    return bin_edges_th1d_x, np.transpose(hist_content_th1d)[0]



