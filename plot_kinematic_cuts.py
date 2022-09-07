#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:21:02 2022

@author: beriksso
"""

import sys
sys.path.insert(0, 'C:/python/definitions/')
import useful_defs as udfs
sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
udfs.set_nes_plot_style()


def import_data(file_name):
    """Import 2D histogram data from given file."""
    p = udfs.unpickle(file_name)
    input_args = p['input_arguments']

    # Projection on t_tof axis
    t_counts = p['counts']

    # Reshape background
    bgr = p['bgr_level']
    t_bgr = np.append(np.flip(bgr[1:]), bgr)

    # 2D matrices
    m_S1 = p['hist2d_S1']
    m_S2 = p['hist2d_S2']

    # Bins
    t_bins = p['bins']
    e_bins_S1 = p['S1_info']['energy bins']
    e_bins_S2 = p['S2_info']['energy bins']

    # Calculate bin centres
    e_S1 = e_bins_S1[1:] - np.diff(e_bins_S1) / 2
    e_S2 = e_bins_S2[1:] - np.diff(e_bins_S2) / 2

    return t_counts, t_bins, t_bgr, e_S1, e_S2, m_S1, m_S2, input_args


def setup_matrix(matrix, x_bins, y_bins):
    """Create fill for x and y."""
    x_repeated = np.tile(x_bins, len(y_bins))
    y_repeated = np.repeat(y_bins, len(x_bins))
    weights = np.ndarray.flatten(np.transpose(matrix))

    return x_repeated, y_repeated, weights


def get_kinematic_cuts(input_arguments, tof):
    """Import equation for kinematic cuts."""
    if '--apply-cut-factors' in input_arguments:
        arg = np.argwhere(input_arguments == '--apply-cut-factors')[0][0]
        c1 = float(input_arguments[arg + 1])
        c2 = float(input_arguments[arg + 2])
        c3 = float(input_arguments[arg + 3])
    else:
        c1 = 1
        c2 = 1
        c3 = 1

    S1_min, S1_max, S2_max = dfs.get_kincut_function(tof, (c1, c2, c3))
    return S1_min, S1_max, S2_max


def plot_for_paper(t_counts, t_bins, t_bgr, e_bins_S1, e_bins_S2, matrix_S1,
                   matrix_S2, inp_args):
    """Plot for technical TOFu paper."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(4, 12)

    # Set colorbar min/max
    vmin = 1
    vmax = (matrix_S1.max() if matrix_S1.max() > matrix_S2.max() else
            matrix_S2.max())
    normed = matplotlib.colors.LogNorm(vmin, vmax)

    # Set white background
    my_cmap = matplotlib.cm.get_cmap('jet').copy()
    my_cmap.set_under('w', 1)

    # Plot S1 2D histogram
    # --------------------
    x_r, y_r, weights = setup_matrix(matrix_S1, t_bins, e_bins_S1)
    ax1.hist2d(x_r, y_r, bins=(t_bins, e_bins_S1), weights=weights,
               cmap=my_cmap, norm=normed)

    # Plot S2 2D histogram
    # --------------------
    x_r, y_r, weights = setup_matrix(matrix_S2, t_bins, e_bins_S2, )
    ax2.hist2d(x_r, y_r, bins=(t_bins, e_bins_S1), weights=weights,
               cmap=my_cmap, norm=normed)

    # Plot TOF projection
    # -------------------
    ax3.plot(t_bins, t_counts, 'k.', markersize=1)
    ax3.errorbar(t_bins, t_counts, np.sqrt(t_counts), color='k',
                 linestyle='None')

    # Plot background component
    ax3.plot(t_bins, t_bgr, 'C0--')

    # Add lines for kinematic cuts
    S1_min, S1_max, S2_max = get_kinematic_cuts(inp_args, t_bins)
    ax1.plot(t_bins, np.array([S1_min, S1_max]).T, 'r')
    ax2.plot(t_bins, S2_max, 'r')

    # Configure plot
    # --------------
    ax1.set_ylabel('$E_{ee}^{S1}$ $(MeV_{ee})$')
    ax2.set_ylabel('$E_{ee}^{S2}$ $(MeV_{ee})$')
    ax3.set_ylabel('counts')
    ax3.set_xlabel('$t_{TOF}$ (ns)')

    y1 = (0, 2.3)
    y2 = (0, 6)
    x3 = (-100, 100)
    y3 = (10, 2E3)
    ax1.set_ylim(y1)
    ax2.set_ylim(y2)
    ax3.set_yscale('log')
    ax3.set_yticks([1, 10, 100, 1000])
    ax3.set_xlim(x3)
    ax3.set_ylim(y3)

    bbox = dict(facecolor='white', edgecolor='black')
    ax1.text(0.89, 0.88, '(a)', transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.89, 0.88, '(b)', transform=ax2.transAxes, bbox=bbox)
    ax3.text(0.89, 0.88, '(c)', transform=ax3.transAxes, bbox=bbox)

    # Add colorbar
    # ------------
    fig.subplots_adjust(top=0.8)
    cbar_ax = fig.add_axes([0.2, 0.84, 0.73, 0.02])
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=normed)
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    plt.subplots_adjust(hspace=0.1)


if __name__ == '__main__':
    plt.close('all')
    # Import data
    f_name = 'data/100850.pickle'
    dat = import_data(f_name)

    # Plot 2d histogram
    plot_for_paper(*dat)

    # Save figure
    plt.savefig('100850_kincut.png', dpi=1200, bbox_inches='tight',
                pad_inches=0)
