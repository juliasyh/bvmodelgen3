#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/01 14:12:16

@author: Javiera Jilberto Vallejos 
'''


import numpy as np
import meshio as io
from tqdm import tqdm

def fourier_time_smoothing(surfaces):
    """
    Parameters:
    surfaces: list of meshio triangular meshes

    Returns:
    modified_surfaces: list of meshio triangular meshes with smoothed points
    """
    print('Smoothing BV surfaces in time...')
    # Load the data
    xyzs = []
    for surface in surfaces:
        xyzs.append(surface.points)

    # Add the first two frames to the end of the list to ensure continuity
    xyzs = [xyzs[-2], xyzs[-1]] + xyzs + [xyzs[0], xyzs[1]]
    xyzs = np.stack(xyzs)

    # Calculate displacements
    disp = xyzs - xyzs[2]
    flatten_disp = disp.reshape(disp.shape[0], -1)
    flatten_disp = np.swapaxes(flatten_disp, 0, 1)

    # Smooth the displacement using the Fourier space
    fourier_transform = np.fft.fft(flatten_disp, axis=1)
    cutoff = 0.2  # Define a cutoff frequency
    freqs = np.fft.fftfreq(fourier_transform.shape[1])
    fourier_transform[:,np.abs(freqs) > cutoff] = 0

    # Inverse Fourier transform to get the smoothed displacement
    smoothed_disp = np.fft.ifft(fourier_transform, axis=1).real
    smoothed_disp_reshape = smoothed_disp.reshape(flatten_disp.shape)

    # Save displacements to mesh
    smoothed_disp_reshape = np.swapaxes(smoothed_disp_reshape, 0, 1)
    smoothed_disp_reshape = smoothed_disp_reshape.reshape(disp.shape)

    modified_surfaces = []
    for i in tqdm(range(2,32)):
        surface = surfaces[i-2]
        surface.points = xyzs[2] + smoothed_disp_reshape[i]
        modified_surfaces.append(surface)
        
    return modified_surfaces