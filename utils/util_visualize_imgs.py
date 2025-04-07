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

def generate_plane(img, affine, pixdim):
    """
    Generate a plane mesh from the given image data.
    
    Parameters:
    img (numpy.ndarray): The image data.
    affine (numpy.ndarray): The affine transformation matrix.
    pixdim (list): The pixel dimensions.
    
    Returns:
    mesh (pyvista.PolyData): The generated plane mesh.
    """
    # Create a plane mesh
    center = (img.shape[0] // 2 - 0.5, img.shape[1] // 2 - 0.5, 0) 
    i_size = (img.shape[0]-1)
    j_size = (img.shape[1]-1)
    mesh = pv.Plane(center=center, i_size=i_size, j_size=j_size, i_resolution=img.shape[0]-1, j_resolution=img.shape[1]-1)
    
    # Set the scalars for the mesh
    mesh.point_data['img'] = img[:,:,0,0].T.flatten()
    
    transformed = mesh.transform(affine, inplace=False)
    
    return mesh, transformed


imgs_fldr = '../test_data/'

la_4ch_img, _, la_4ch_pixdim = readFromNIFTI(f'{imgs_fldr}/LA_4CH')
la_3ch_img, _, la_3ch_pixdim = readFromNIFTI(f'{imgs_fldr}/LA_3CH')
la_2ch_img, _, la_2ch_pixdim = readFromNIFTI(f'{imgs_fldr}/LA_2CH')
la_4ch_seg, la_4ch_affine, la_4ch_pixdim = readFromNIFTI(f'{imgs_fldr}/la_4ch_seg')
la_3ch_seg, la_3ch_affine, la_3ch_pixdim = readFromNIFTI(f'{imgs_fldr}/la_3ch_seg')
la_2ch_seg, la_2ch_affine, la_2ch_pixdim = readFromNIFTI(f'{imgs_fldr}/la_2ch_seg')

la_4ch_mesh, la_4ch_mesh_t = generate_plane(la_4ch_img, la_4ch_affine, la_4ch_pixdim)
la_3ch_mesh, la_3ch_mesh_t = generate_plane(la_3ch_img, la_3ch_affine, la_3ch_pixdim)
la_2ch_mesh, la_2ch_mesh_t = generate_plane(la_2ch_img, la_2ch_affine, la_2ch_pixdim)

plotter = pv.Plotter()
plotter.add_mesh(la_4ch_mesh_t, scalars='img', show_edges=False)
plotter.add_mesh(la_3ch_mesh_t, scalars='img', show_edges=False)
plotter.add_mesh(la_2ch_mesh_t, scalars='img', show_edges=False)
plotter.export_html(f'{imgs_fldr}/images.html')
