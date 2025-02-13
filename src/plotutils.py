#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:07:18 2022

@author: Javiera Jilberto Vallejos
"""
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import  plot


def show_point_cloud(points, fig=None, color=None, size=10, cmap='Viridis',
                     opacity=1, marker_symbol='circle', label=None, showscale=False,
                     cmin = None, cmax = None):
    if fig is None:
        fig = go.Figure()
    if len(points.shape) == 1:
        points = points[None]
    fig.add_scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers',
                      marker=dict(
                                    color=color,
                                    size=size,
                                    colorscale=cmap,
                                    opacity=opacity,
                                    symbol=marker_symbol,
                                    showscale=showscale,
                                    cmin = cmin,
                                    cmax = cmax
                                ),
                      name = label
        )
    return fig


def save_figure(fname, fig):
    fig.write_html(fname)


def plot_slices(slices, show=False, which='lv', lge=False):
    fig = go.Figure()
    for n in range(len(slices)):
        if slices[n].view == 'sa':
            color = 'blue'
        else:
            color = 'red'
        xyz = slices[n].get_xyz_trans(which)
        if lge:
            fig = show_point_cloud(xyz[slices[n].lge_data==2], opacity=0.5, size=5, color='blue', fig=fig,
                                      label=(slices[n].view+str(slices[n].slice_number)))
            fig = show_point_cloud(xyz[slices[n].lge_data==1], opacity=0.5, size=5, color='red', fig=fig,
                                      label=(slices[n].view+str(slices[n].slice_number)))
        else:
            fig = show_point_cloud(xyz, opacity=0.5, size=5, color=color, fig=fig,
                                  label=(slices[n].view+str(slices[n].slice_number)))

    if show:
        fig.show()
    return fig


def plot_contours(contours, background=True):
    colors = {'lvendo': 'blue', 'lvepi': 'red', 'rvendo': 'green', 'rvsep': 'cyan',
              'rvinsert': 'black', 'mv': 'yellow', 'av': 'purple', 'tv': 'magenta',
              'apexendo': 'black', 'apexepi': 'black', 'rvapex': 'black'}
    sizes = {'lvendo': 5, 'lvepi': 5, 'rvendo': 5, 'rvsep': 5,
              'rvinsert': 10, 'mv': 10, 'av': 10, 'tv': 10,
              'apexendo': 10, 'apexepi': 10, 'rvapex': 10}
    ctype = {}
    for ctr in contours:
        try:
            ctype[ctr.ctype].append(ctr.points)
        except:
            ctype[ctr.ctype] = [ctr.points]

    for key in ctype.keys():
        ctype[key] = np.vstack(ctype[key])

    fig = go.Figure()
    for key in ctype.keys():
        show_point_cloud(ctype[key], fig=fig, opacity=0.9, color=colors[key], label=key, size=sizes[key])

    if background:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    return fig

def contours2vertex(contours):
    import meshio as io
    ctype = {}
    for ctr in contours:
        try:
            ctype[ctr.ctype].append(ctr.points)
        except:
            ctype[ctr.ctype] = [ctr.points]

    for key in ctype.keys():
        ctype[key] = np.vstack(ctype[key])

    points = []
    label = []
    for i, key in enumerate(ctype.keys()):
        points.append(ctype[key])
        label.append([i]*len(ctype[key]))

    points = np.vstack(points)
    label = np.concatenate(label)
    mesh = io.Mesh(points, {'vertex': np.arange(len(points))[:,None]}, point_data={'label': label})
    return mesh

def plot_surface(model, contourPlots=None, out_path=None):
    if contourPlots is None:
        data = model
    else:
        data = model + contourPlots

    fig = go.Figure(data)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    if out_path is not None:
        plot(fig,filename=out_path, auto_open=False)
    return fig