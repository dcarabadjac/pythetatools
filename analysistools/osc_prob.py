# -*- coding: utf-8 -*-

from .default_parameters import *
from .hamiltonians import *
from numpy import *
import numpy as np
import cmath
import cmath as cmath

from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt 

def K(U, beta, i, j):
    return U[1][i]*conj(U[beta][i])*conj(U[1][j])*U[beta][j]


def PMNS(theta_12=theta_12_BF, theta_13=theta_13_BF, theta_23=theta_23_BF,\
         delta_cp=delta_cp_BF):
    r"""Returns PMNS matrix

    Parameters
    ----------
    3 mixing angles and delta_cp phase
    By default best fit values are set from default_parameters file

    Returns
    -------
    numpy array
        2D numpy array of the PMNS matrix
    """
    s13 = sin(theta_13)
    c13 = cos(theta_13)
    s23 = sin(theta_23)
    c23 = cos(theta_23)
    s12 = sin(theta_12)
    c12 = cos(theta_12)

    s2_13 = sin(theta_13)**2
    c2_13 = cos(theta_13)**2
    s2_23 = sin(theta_23)**2
    c2_23 = cos(theta_23)**2
    c2_12 = cos(theta_12)**2
    s2_12 = sin(theta_12)**2

    A = np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
    B = np.array([[c13, 0, s13*exp(complex(0,-delta_cp))], [0, 1, 0], [-s13*exp(complex(0,delta_cp)), 0, c13]])
    C = np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])


    U = np.dot(np.dot(A, B), C)
    return U


def cronecer(beta):
    if beta==1:
        return 1
    else:
        return 0
    
    
def probability_vacuum(U, beta, E, L=baseline, dm2_21=dm2_21_BF, dm2_32=dm2_32_BF):
    r"""The probability of oscillations of muon neutrino for different energies in vacuum

    Parameters
    ----------
    U - PMNS matrix
    beta - channel of the oscillations (beta=0 - electronic, beta=1 - muonic channel, beta=2 - taonic channel)
    E - energy of oscillating beam
    L - baseline of the oscillations

    Returns
    -------
    numpy array
        probability of oscillations into given channel beta, at given energy E
    """
    
    dm2_31 = dm2_32 + dm2_21
    Delta_21 = 1.2669*dm2_21*L/E
    Delta_32 = 1.2669*dm2_32*L/E
    Delta_31 = 1.2669*dm2_31*L/E
    sin2_21 = sin(Delta_21)**2
    sin2_32 = sin(Delta_32)**2
    sin2_31 = sin(Delta_31)**2

    sind_21 = sin(2*Delta_21)
    sind_32 = sin(2*Delta_32)
    sind_31 = sin(2*Delta_31)

    cosd_21 = cos(2*Delta_21)
    cosd_32 = cos(2*Delta_32)
    cosd_31 = cos(2*Delta_31)

    return (cronecer(beta) - 4*(K(U, beta, 1, 0)*sin2_21 + K(U, beta, 2, 0)*sin2_31\
           + K(U, beta, 2, 1)*sin2_32).real - 2*(K(U, beta, 1, 0)*sind_21\
           + K(U, beta, 2, 0)*sind_31+ K(U, beta, 2, 1)*sind_32).imag)


def probability_vacuum_approx(E, beta, theta_13=theta_13_BF, theta_23=theta_23_BF,\
           L=baseline, dm2_32=dm2_32_BF):
    r"""Computes the leading approximation of the probability of oscillations of muon neutrino in the long-baseline experiment

    Parameters
    ----------
    beta - channel of the oscillations (beta=0 - electronic, beta=1 - muonic channel, beta=2 - taonic channel)
    E - energy of oscillating beam
    L - baseline of the oscillations

    Returns
    -------
    numpy array
        approximated probability of oscillations into given channel beta, at given energy E
    """
    Delta_32 = 1.2669*dm2_32*L/E
    sin2_32 = sin(Delta_32)**2
    
    if beta==0:
        P = sin(2*theta_13)**2*sin(theta_23)**2*sin(Delta_32)**2
    elif beta==1:
        P = 1 - sin(2*theta_23)**2*cos(theta_13)**4*sin(Delta_32)**2
    return P



def probability_vacuum_anti(U, beta, E, L=baseline, dm2_21=dm2_21_BF, dm2_32=dm2_32_BF):
    r"""The probability of oscillations of antimuon neutrino for different energies in vacuum

    Parameters
    ----------
    U - PMNS matrix
    beta - channel of the oscillations (beta=0 - electronic, beta=1 - muonic channel, beta=2 - taonic channel)
    E - energy of oscillating beam
    L - baseline of the oscillations

    Returns
    -------
    numpy array
        probability of antimuon oscillations into given channel beta, at given energy E
    """
    
    dm2_31=dm2_21+dm2_32
    Delta_21 = 1.2669*dm2_21*L/E
    Delta_32 = 1.2669*dm2_32*L/E
    Delta_31 = 1.2669*dm2_31*L/E

    sin2_21 = pow(sin(Delta_21),2.0)
    sin2_32 = pow(sin(Delta_32),2.0)
    sin2_31 = pow(sin(Delta_31),2.0)

    sind_21 = sin(2*Delta_21)
    sind_32 = sin(2*Delta_32)
    sind_31 = sin(2*Delta_31)

    cosd_21 = cos(2*Delta_21)
    cosd_32 = cos(2*Delta_32)
    cosd_31 = cos(2*Delta_31)

    return (cronecer(beta) - 4*(K(U, beta, 1, 0)*sin2_21 + K(U, beta, 2, 0)*sin2_31\
           + K(U, beta, 2, 1)*sin2_32).real + 2*(K(U, beta, 1, 0)*sind_21\
           + K(U, beta, 2, 0)*sind_31+ K(U, beta, 2, 1)*sind_32).imag)




def probability_general(E, beta, theta_12=theta_12_BF,  theta_13=theta_13_BF, theta_23=theta_23_BF,delta_cp=delta_cp_BF,    dm2_21=dm2_21_BF,dm2_atm=dm2_32_BF, compute_matrix_multiplication=True, L=baseline*CONV_KM_TO_INV_EV, ME=True, MO='NO'):
    r"""The probability of oscillations of muon neutrino for different energies including matter effects (SU(3) expansion method is used)

    Parameters
    ----------
    all fundamenta oscillation parameters - by default BF values are set
    beta - channel of the oscillations (beta=0 - electronic, beta=1 - muonic channel, beta=2 - taonic channel)
    E - energy of oscillating beam
    L - baseline of the oscillations
    ME - matter effects (if ME=True, ME are included, if ME=False, ME will be ignored (oscillations in vacuum))
    Returns
    -------
    numpy array
        the most general and exact probability of muon oscillations into given channel beta, at given energy E
    """
    if MO=='NO':
        dm2_32 = dm2_atm
    if MO=='IO':
        dm2_32 = -dm2_atm - dm2_21
    
    if ME:
        VCC = VCC_EARTH_CRUST
    
        h_vacuum_energy_ind = hamiltonian_3nu_energy_independent(theta_12,  theta_13, theta_23,\
                                                  delta_cp, dm2_21, dm2_32,\
                                                  compute_matrix_multiplication=True)
        VCC_matrix = [[VCC, 0, 0], [0, 0, 0], [0, 0, 0]]
        return np.array([probability_energy_fixed(np.multiply(1./x/1.e9, h_vacuum_energy_ind) + VCC_matrix, beta, L) for x in E])    
    else:
        U = PMNS(theta_12=theta_12, theta_13=theta_13, theta_23=theta_23,\
         delta_cp=delta_cp)
        return probability_vacuum(U=U, beta=beta, E=E, L=baseline, dm2_21=dm2_21, dm2_32=dm2_32)

def probability_general_anti(E, beta, theta_12=theta_12_BF,  theta_13=theta_13_BF, theta_23=theta_23_BF,delta_cp=delta_cp_BF, dm2_21=dm2_21_BF, dm2_atm=dm2_32_BF, compute_matrix_multiplication=True, L=baseline*CONV_KM_TO_INV_EV, ME=True, MO="NO"):
    r"""See probability_general. Just for antineutrino
    """
    
    if MO=='NO':
        dm2_32 = dm2_atm
    if MO=='IO':
        dm2_32 = -dm2_atm - dm2_21
    
    if ME:
        VCC = -VCC_EARTH_CRUST
    
        h_vacuum_energy_ind = hamiltonian_3nu_energy_independent(theta_12,  theta_13, theta_23,\
                                                  -delta_cp, dm2_21, dm2_32,\
                                                  compute_matrix_multiplication=True)
        VCC_matrix = [[VCC, 0, 0], [0, 0, 0], [0, 0, 0]]
        return np.array([probability_energy_fixed(np.multiply(1./x/1.e9, h_vacuum_energy_ind) + VCC_matrix, beta, L) for x in E])    
    else:
        U = PMNS(theta_12=theta_12, theta_13=theta_13, theta_23=theta_23,\
         delta_cp=delta_cp)
        return probability_vacuum_anti(U=U, beta=beta, E=E, L=baseline, dm2_21=dm2_21, dm2_32=dm2_32)
    
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc
