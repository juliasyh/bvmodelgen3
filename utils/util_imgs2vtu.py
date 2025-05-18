#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/02 09:12:58

@author: Javiera Jilberto Vallejos 
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import meshio as io
import pyvista as pv
from scipy.spatial.transform import Rotation

from niftiutils import readFromNIFTI

from tqdm import tqdm


def generate_plane(img, affine, translation=np.array([0,0]), slice=0, time_frame=0):
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
    img_slice = img[:, :, slice, time_frame]
    ij = mesh.points[:, :2].astype(int)
    ij[:,0] = ij[:,0] - translation[0]
    ij[:,1] = ij[:,1] - translation[1]
    ij[:,0] = np.clip(ij[:,0], 0, img.shape[0]-1)
    ij[:,1] = np.clip(ij[:,1], 0, img.shape[1]-1)
    img_slice = img_slice[ij[:, 0], ij[:, 1]]

    mesh.point_data['img'] = img_slice
    mesh.points[:, 2] = slice
    
    transformed = mesh.transform(affine, inplace=False)
    
    return mesh, transformed


imgs_fldr = '/Users/jjv/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/Desmoplakin/Models/DSPPatients2/DSP-11/Images/'
img2model_fldr = f'{imgs_fldr}/img2model/'
if not os.path.exists(f'{imgs_fldr}/img_vtus/'):
    os.makedirs(f'{imgs_fldr}/img_vtus/')

# Create dictionaries to store image data, affine matrices, and pixel dimensions
img_paths = {
    'la_4ch': f'{imgs_fldr}/LA_4CH',
    'la_3ch': f'{imgs_fldr}/LA_3CH',
    'la_2ch': f'{imgs_fldr}/LA_2CH',
    'sa': f'{imgs_fldr}/SA'
}

seg_paths = {
    'la_4ch': f'{imgs_fldr}/la_4ch_seg',
    'la_3ch': f'{imgs_fldr}/la_3ch_seg',
    'la_2ch': f'{imgs_fldr}/la_2ch_seg',
    'sa': f'{imgs_fldr}/sa_seg'
}

imgs = {}
segs = {}
affines = {}

for key in img_paths.keys():
    imgs[key], _, _ = readFromNIFTI(img_paths[key])
    segs[key], affines[key], _ = readFromNIFTI(seg_paths[key])

for key in img_paths.keys():
    img = imgs[key]
    affine = affines[key]
    slice_index = 5 if key == 'sa' else 0  # Use slice=10 for 'sa', otherwise slice=0

    # Load translations
    translations = np.load(f'{img2model_fldr}/frame{0}_{key.lower()}_translations.npy')
    translations = np.round(translations).astype(int)

    for ts in tqdm(range(img.shape[3])):
        _, transformed_mesh = generate_plane(img, affine, translation=translations[slice_index], 
                                             slice=slice_index, time_frame=int(ts))
        pv.save_meshio(f'{imgs_fldr}/img_vtus/{key}_slice{slice_index}_time{ts}.vtu', transformed_mesh)
