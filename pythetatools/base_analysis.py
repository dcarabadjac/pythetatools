import numpy as np
from scipy.stats import chi2, norm, chi2
import scipy.stats as stats
from pathlib import Path


def marg_mean(x1, x2, prof=False):
    if prof:
        return np.minimum(x1, x2)
    return -np.log((np.exp(-x1)/2+np.exp(-x2)/2))

def get_double_sided_gaussian_zscore(x):
    return np.round(norm.ppf((1-x/2)), 4)

def quadr(x, a, b, c):
    return a*x**2 + b*x + c

def find_parabold_vertex(x1, x2, x3, y1, y2, y3):
    h = x3-x2
    a = ((y3+y1)/2. - y2)/(h*h)
    b = ((y3-y1)/2. - 2.*a*h*x2)/h
    c = y2 - a*x2**2 - b*x2
    xmin = -b/(2.*a)
    return xmin, quadr(xmin, a, b, c)

def poisson_error_bars(Nobs, alpha):
    yerr_low = np.zeros_like(Nobs, dtype=float)
    yerr_high = np.zeros_like(Nobs, dtype=float)

    for i, n in enumerate(Nobs):
        if n == 0:
            yerr_low[i] = 0
        else:
            yerr_low[i] = n - stats.gamma.ppf(alpha / 2, n, scale=1)
        
        yerr_high[i] = stats.gamma.ppf(1 - alpha / 2, n + 1, scale=1) - n

    return yerr_low, yerr_high

def divide_arrays(array_nom, array_denom):
    where_zeros = array_denom==0
    ratio = np.zeros_like(array_nom, dtype=float)
    ratio[where_zeros] = 0
    ratio[~where_zeros] = array_nom[~where_zeros] / array_denom[~where_zeros]
    return ratio


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
    
def sigma_to_CL(nsigma):
    return round(chi2.cdf(nsigma**2, 1), 4)

def CL_to_chi2critval(CL, dof):
    """Calculate the Δχ² value for a given confidence level and degrees of freedom."""
    return chi2.ppf(CL, dof)


def find_intersections(x1, y1, x2, y2):
    y2_interp = np.interp(x1, x2, y2)

    diff = y1 - y2_interp

    idx = np.where(np.diff(np.sign(diff)))[0]

    x_intersections = x1[idx] - diff[idx] * (x1[idx+1] - x1[idx]) / (diff[idx+1] - diff[idx])
    y_intersections = y1[idx] + (y1[idx+1] - y1[idx]) * (x_intersections - x1[idx]) / (x1[idx+1] - x1[idx])

    return np.round(x_intersections, 3)

