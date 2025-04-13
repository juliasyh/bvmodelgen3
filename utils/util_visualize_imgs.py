#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/02 09:12:58

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt
import meshio as io
import pyvista as pv
from scipy.spatial.transform import Rotation

from niftiutils import readFromNIFTI

import nibabel as nib


def generate_plane(img, affine, slice=0, time_frame=0):
    """
    Generate a plane mesh from the given image data.
    
    Parameters:
    img (numpy.ndarray): The image data.
    affine (numpy.ndarray): The affine transformation matrix.
    slice (int): The slice index.
    time_frame (int): The time frame index.
    
    Returns:
    mesh (pyvista.PolyData): The generated plane mesh.
    """
    # Create a plane mesh
    center = (img.shape[0] // 2 - 0.5, img.shape[1] // 2 - 0.5, 0) 
    i_size = (img.shape[0]-1)
    j_size = (img.shape[1]-1)
    mesh = pv.Plane(center=center, i_size=i_size, j_size=j_size, i_resolution=img.shape[0]-1, j_resolution=img.shape[1]-1)
    
    # Set the scalars for the mesh
    mesh.point_data['img'] = img[:, :, slice, time_frame].T.flatten()
    mesh.points[:, 2] = slice
    
    transformed = mesh.transform(affine, inplace=False)
    
    return mesh, transformed


imgs_fldr = '../test_data/Images/'

la_4ch_img, _, la_4ch_pixdim = readFromNIFTI(f'{imgs_fldr}/LA_4CH')
la_3ch_img, _, la_3ch_pixdim = readFromNIFTI(f'{imgs_fldr}/LA_3CH')
la_2ch_img, _, la_2ch_pixdim = readFromNIFTI(f'{imgs_fldr}/LA_2CH')
sa_img, _, sa_pixdim = readFromNIFTI(f'{imgs_fldr}/SA')

la_4ch_seg, la_4ch_affine, la_4ch_pixdim = readFromNIFTI(f'{imgs_fldr}/la_4ch_seg')
la_3ch_seg, la_3ch_affine, la_3ch_pixdim = readFromNIFTI(f'{imgs_fldr}/la_3ch_seg')
la_2ch_seg, la_2ch_affine, la_2ch_pixdim = readFromNIFTI(f'{imgs_fldr}/la_2ch_seg')
sa_seg, sa_affine, sa_pixdim = readFromNIFTI(f'{imgs_fldr}/sa_seg')

# Initialize the plotter with anti-aliasing enabled
plotter = pv.Plotter(off_screen=False)
plotter.enable_anti_aliasing()

# Callback function to update the meshes based on the slider value
def update_time_frame(time_frame):
    _, la_4ch_mesh_t = generate_plane(la_4ch_img, la_4ch_affine, time_frame=int(time_frame))
    _, la_3ch_mesh_t = generate_plane(la_3ch_img, la_3ch_affine, time_frame=int(time_frame))
    _, la_2ch_mesh_t = generate_plane(la_2ch_img, la_2ch_affine, time_frame=int(time_frame))
    _, sa_mesh_t = generate_plane(sa_img, sa_affine, slice=10, time_frame=int(time_frame))
    
    plotter.add_mesh(la_4ch_mesh_t, scalars='img', show_edges=False)
    plotter.add_mesh(la_3ch_mesh_t, scalars='img', show_edges=False)
    plotter.add_mesh(la_2ch_mesh_t, scalars='img', show_edges=False)
    plotter.add_mesh(sa_mesh_t, scalars='img', show_edges=False)
    plotter.render()

# Add a slider to control the time frame and position it on the right
plotter.add_slider_widget(update_time_frame, 
                          rng=(0, la_4ch_img.shape[3] - 1), 
                          value=0, 
                          title="Time Frame", 
                          style='modern', 
                          pointa=(0.9, 0.1), 
                          pointb=(0.9, 0.9))

# Initial rendering
update_time_frame(0)

# Show the plotter
plotter.export_html(f'{imgs_fldr}/images.html')
plotter.show()