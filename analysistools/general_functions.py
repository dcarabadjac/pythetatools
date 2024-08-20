from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.transforms as transforms


    

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
    if  axis=='both':
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=6, width=0.8)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=6, width=0.8)
    elif axis=='x':
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=6, width=0.8)
    elif axis=='y':
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=6, width=0.8)
    else:
        print("Unknown axis")

def avnllh_to_dchi2(avnllh, min_value):
    return 2*(avnllh-min_value)

def shift_bbox_center(bbox, dx, dy):
    # Calculate current center
    current_center = bbox.get_points().mean(axis=0)

    # Calculate new center
    new_center = current_center + [dx, dy]

    # Calculate shift amounts
    shift_x = new_center[0] - current_center[0]
    shift_y = new_center[1] - current_center[1]

    # Create a transform to shift the bbox
    translate = transforms.Affine2D().translate(shift_x, shift_y)

    # Apply the transform to the bbox
    bbox_transformed = bbox.transformed(translate)

    return bbox_transformed
