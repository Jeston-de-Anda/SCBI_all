import torch
import stability_testers as st
import numpy as np
import sinkhorn_torch as sk

import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import prior_variation
pv = prior_variation.PriorVariation

def parse_data(data, layers=[], pv=pv, hypo=0):
    """
    Parse Data
    ----------
    Transform raw data to numpy 2d-array of lattice-base coordinates,
    and read the color value in the same order.
    
    Parameters
    ----------
    data  : raw data returned by `PriorVariation.read_data`
    layers: *mutable* iterable, containing 0 (the center) and 
            other layer labels in data (no checking, input carefully).
            `layers = []` means all existing layers with center.
    pv    : class `PriorVariation`
    hypo  : hypothesis, default 0
    
    Return
    ------
    (points, color)
    points: numpy 2d-array of shape (n,2), coordinates in lattice-base coordinantes
    color : numpy 1d-array of shape (n,), color values at each point.
    """
    if not layers:
        layers = data["base"].keys()
        pts = [np.array([[0,0]]),]
        color = [np.array([1,]), ]
    else:
        layers = list(layers)
        pts = []
        color = []

    if 0 in layers:
        pts += [np.array([[0,0]]),]
        color += [np.array([1,]), ]
        layers.remove(0)
    
    for key in layers:
        array = data["base"][key]
        pts += [array, ]
        array = data["learn"][key]
        color += [np.array(array)[:,0],]
        
    pts = np.concatenate(pts, axis=0)
    color = np.concatenate(color, axis=0)
    
    return pts, color

def base2coord(pts, target="visual", setup=None, pv=pv):
    """
    Coordinate transformation: base to target
    
    Parameters
    ----------
    pts   : numpy 2d-array of shape (n,dim), points in lattice-base coordinate
    target: visual or simplex, string of first 1, 3, or full letters
    setup : raw setup returned by `PriorVariation.read_data`
    pv    : the class `PriorVariation`
    
    Return
    ------
    numpy 2d-array of shape (n, target_dim), coordinates of points, respectively
    """
    choice={"v":pv.trans_visual,
            "vis":pv.trans_visual,
            "visual":pv.trans_visual,
            "s":pv.trans_simplex,
            "sim":pv.trans_simplex,
            "simplex":pv.trans_simplex,
           }
    pts = pts.reshape([-1,pts.shape[-1]]) # if shape is not coorect.
    
    if setup is None:
        dim = 3
        method = "hex"
        # density = 6
        center = np.ones(3)/3
        resolution = 0.01
    else:
        dim = setup["matrix"].shape[1]
        method = setup["method"]
        # density = setup["density"]
        center = setup["prior_t"].numpy()
        resolution = setup["resolution"]

    if "v" in target:
        center -= np.ones(3)/3
        center = np.matmul(center[:dim-1], np.linalg.inv(choice["simplex"][dim][:,:dim-1]))
        center = np.matmul(center, choice["visual"][dim])
        
    trans_mat = choice[target][dim]
    
    coords = np.matmul(pts, trans_mat)
    coords = coords * resolution + center
    return coords

def base2vis(pts, setup=None, pv=pv):
    return base2coord(pts,"visual", setup, pv)

def base2sim(pts, setup=None, pv=pv):
    return base2coord(pts,"simplex", setup, pv)


def graph3d(data, color=None, log=True, simplex=False,  markersize=13):
    """
    Graph3d
    
    Draw scatter graphs given by data with colors given by color.
    data is a numpy 2d-array of shape (n,3)
    """
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if simplex:
        ax.plot3D([1,0,0,1],[0,1,0,0],[0,0,1,0])
    
    if color is not None:
        if log:
            s = ax.scatter(data[:,0], data[:,1], data[:, 2], c=1-color, 
                           norm=colors.LogNorm(vmin=(1-color).min(), vmax=(1-color).max()),
                           cmap = mpl.cm.get_cmap("viridis_r"),
                           marker='h', )
        else:
            s = ax.scatter(data[:,0], data[:,1], data[:, 2], c=color, marker='h',  )
        fig.colorbar(s,ax=ax)
    else:
        ax.scatter(data[:,0], data[:,1], data[:, 2], marker='h',  )
        
    ax.legend()
    
    
    return fig, ax

def graph2d(data, color=None, log=True, simplex=False, markersize=13):
    """
    Graph2d
    """
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca()
    
    fr = np.array([[0,1], [-np.sqrt(3)/2,-1/2,], [np.sqrt(3)/2,-1/2],[0,1]])
    if simplex:
        ax.plot(fr[:, 0], fr[:, 1])
    
    if color is not None:
        if log:
            s = ax.scatter(data[:,0], data[:,1], c=1-color, 
                           norm=colors.LogNorm(vmin=(1-color).min(), vmax=(1-color).max()),
                           cmap = mpl.cm.get_cmap("viridis_r"), marker='h', )
        else:
            s = ax.scatter(data[:,0], data[:,1], c=color, marker='h', )
        fig.colorbar(s,ax=ax)
    else:
        ax.scatter(data[:,0], data[:,1], marker='h', )
        
    ax.legend()
    
    return fig, ax