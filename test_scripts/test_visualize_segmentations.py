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
    mesh.point_data['scalars'] = img[:,:,0,0].T.flatten()
    
    transformed = mesh.transform(affine, inplace=False)
    
    return mesh, transformed

# TODOs
# - Figure out how to save the mesh to a file
# - Load temporal 
# - Add translations

la_4ch_img, _, la_4ch_pixdim = readFromNIFTI('../test_data/Images/LA_4CH')
la_3ch_img, _, la_3ch_pixdim = readFromNIFTI('../test_data/Images/LA_3CH')
la_2ch_img, _, la_2ch_pixdim = readFromNIFTI('../test_data/Images/LA_2CH')
la_4ch_seg, la_4ch_affine, la_4ch_pixdim = readFromNIFTI('../test_data/Images/la_4ch_seg')
la_3ch_seg, la_3ch_affine, la_3ch_pixdim = readFromNIFTI('../test_data/Images/la_3ch_seg')
la_2ch_seg, la_2ch_affine, la_2ch_pixdim = readFromNIFTI('../test_data/Images/la_2ch_seg')

la_4ch_mesh, la_4ch_mesh_t = generate_plane(la_4ch_img, la_4ch_affine, la_4ch_pixdim)
la_3ch_mesh, la_3ch_mesh_t = generate_plane(la_3ch_img, la_3ch_affine, la_3ch_pixdim)
la_2ch_mesh, la_2ch_mesh_t = generate_plane(la_2ch_img, la_2ch_affine, la_2ch_pixdim)


plotter = pv.Plotter()
plotter.add_mesh(la_4ch_mesh_t, scalars='scalars', show_edges=True)
plotter.add_mesh(la_3ch_mesh_t, scalars='scalars', show_edges=False)
plotter.add_mesh(la_2ch_mesh_t, scalars='scalars', show_edges=False)
plotter.show()

# #%% Test with 4CH
# la_4ch_ijk = la_4ch_mesh.points[::20]
# la_4ch_xyz = nib.affines.apply_affine(la_4ch_affine, la_4ch_ijk)

# la_2ch_ijk = la_2ch_mesh.points[::20]
# la_2ch_xyz = nib.affines.apply_affine(la_2ch_affine, la_2ch_ijk)


# # Create a 3D scatter plot of la_4ch_xyz and la_2ch_xyz
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot la_4ch_xyz
# ax.scatter(la_4ch_xyz[:, 0], la_4ch_xyz[:, 1], la_4ch_xyz[:, 2], c='r', label='LA 4CH', alpha=0.6)

# # Plot la_2ch_xyz
# ax.scatter(la_2ch_xyz[:, 0], la_2ch_xyz[:, 1], la_2ch_xyz[:, 2], c='b', label='LA 2CH', alpha=0.6)

# # Set labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# ax.set_aspect('equal')

# # Show the plot
# plt.show()

#%%

img, img_affine, img_pixdim = readFromNIFTI('../test_data/Images/LA_4CH')
seg, seg_affine, seg_pixdim = readFromNIFTI('../test_data/Images/la_4ch_seg')

origin = np.array([0, 0, 0])
xvector = np.array([1, 0, 0])

print(np.linalg.norm(nib.affines.apply_affine(img_affine, xvector) - nib.affines.apply_affine(img_affine, origin)))
print(np.linalg.norm(nib.affines.apply_affine(seg_affine, xvector) - nib.affines.apply_affine(seg_affine, origin)))

