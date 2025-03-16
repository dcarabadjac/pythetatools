from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import Bbox
from collections import defaultdict
from .base_analysis import poisson_error_bars
from copy import copy


def plot_stacked_samples(ax, samples, labels, **kwargs):
    """
    Plots a 1D histogram (like in ROOT).

    Parameters
    ----------
    samples : list
        The list of the histograms to be stacked
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `ax.step` function 
        for customizing the plot (e.g., color, linewidth, linestyle).
    """
    # Define default styles
    default_kwargs = {
        'alpha': 1.0,
        'step': 'pre'
    }
    default_kwargs.update(kwargs)
    
    colors = [ "#4A6FA5", "#2A9D8F", "#E9C46A", "#E76F51", "#A88FDC", "#F4A261"]
    
    hist_stacked_prev = copy(samples[0])
    hist_stacked_prev.set_z(hist_stacked_prev.z*0)
    x = hist_stacked_prev.bin_edges[0]
    for label, sample, color in zip(labels, samples, colors):
        hist_stacked_current = sample + hist_stacked_prev
        ax.fill_between(x, np.insert(hist_stacked_prev.z, 0, 0), np.insert(hist_stacked_current.z, 0, 0), zorder=0, label=label, color=color, **default_kwargs)
        hist_stacked_current.plot(ax, linewidth=0)
        hist_stacked_prev = hist_stacked_current
        

def plot_histogram(ax, xedges, z, rotate=False, **kwargs):
    """
    Plots a 1D histogram (like in ROOT).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the histogram.
    xedges : numpy array
        The bin edges along the x-axis.
    z : numpy array
        The histogram bin heights corresponding to the bin edges.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `ax.step` function 
        for customizing the plot (e.g., color, linewidth, linestyle).

    Notes
    -----
    - This function uses a step plot to visualize the histogram, extending 
      the first and last bin heights to zero for proper display.
    - The length of `z` should be one less than the length of `xedges` 
      because the bin heights correspond to the intervals between bin edges.
    """
    x = np.append(xedges, xedges[-1:])
    heights = np.insert(z, 0, 0)
    heights = np.append(heights, [0])
    if not rotate:
        ax.step(x, heights, **kwargs)
    else:
        ax.invert_yaxis()
        ax.step(-heights, x, where='post', **kwargs)


def plot_data(ax, xedges, z, yerr=None, rotate=False, label=None):
    x_centers = (xedges[1:] + xedges[:-1])/2
    bin_widths = (xedges[1:] - xedges[:-1])/2

    if yerr is None:
        alpha = 1 - 0.6827
        yerr_low, yerr_high = poisson_error_bars(z, alpha)
        yerr = (yerr_low, yerr_high)
    if not rotate:
        ax.errorbar(x_centers, z, yerr=yerr, xerr=bin_widths, fmt='o', label=label, color='black', markersize=5, capsize=2, elinewidth=2, capthick=1)
    else:
        ax.errorbar(-z, x_centers, yerr=bin_widths, xerr=(yerr[1], yerr[0]), fmt='o', label=label, color='black', markersize=5, capsize=2, elinewidth=2, capthick=1)


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


def shift_bbox_center(fig, ax, dx, dy, exp):
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


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    #items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)