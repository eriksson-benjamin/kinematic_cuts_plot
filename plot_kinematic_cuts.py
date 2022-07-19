#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:21:02 2022

@author: beriksso
"""

import useful_defs as udfs
import sys
sys.path.insert(0, '/home/beriksso/TOFu/analysis/benjamin/github/TOFu/functions')
import tofu_functions as dfs
import numpy as np
import matplotlib.pyplot as plt

def import_data(file_name):
    '''
    Import 2D histogram data from given file
    '''
    p = udfs.unpickle(file_name)
    t_counts = p['counts']
    t_bins = p['bins']
    e_bins_S1 = p['S1_info']['energy bins']
    e_bins_S2 = p['S2_info']['energy bins']
    h_S1 = p['hist2d_S1']
    h_S2 = p['hist2d_S2']
    
    return t_counts, t_bins, e_bins_S1, e_bins_S2, h_S1, h_S2

def plot_matrix(matrix, bin_centres, log=False, xlabel=None, ylabel=None):
    plt.figure()
    if bin_centres==None:
        x_bins = np.arange(0, np.shape(matrix)[0])
        y_bins = np.arange(0, np.shape(matrix)[1])
    else:
        x_bins = bin_centres[0]
        y_bins = bin_centres[1]
    
    # Create fill for x and y
    x_repeated = np.tile(x_bins, len(y_bins))
    y_repeated = np.repeat(y_bins, len(x_bins))
    weights = np.ndarray.flatten(np.transpose(matrix))
    
    # Set white background
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)
    
    if log: normed = matplotlib.colors.LogNorm(vmin=1)
    else: normed = None
    # Create 2D histogram using weights
    hist2d = plt.hist2d(x_repeated, y_repeated, bins=(x_bins, y_bins), 
                        weights=weights, cmap=my_cmap, norm=normed)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    return hist2d


# Import data
file_name = 'data/98044_47.3_56.2.pickle'
t_counts, t_bins, e_bins_S1, e_bins_S2, h_S1, h_S2 = import_data(file_name)



