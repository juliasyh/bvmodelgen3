#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/14 17:36:13

@author: Javiera Jilberto Vallejos 
'''

import os

import numpy as np
import pandas as pd
import meshio as io

from scipy.spatial import KDTree

import tetgen

# Reading surface control mesh
control_mesh_dir = "../"
filemod='_mod'

model_file = os.path.join(control_mesh_dir,"model" + filemod + ".txt")
if not os.path.exists(model_file):
    ValueError('Missing model.txt file')
control_mesh = (pd.read_table
                        (model_file, sep='\s+', header=None)).values

subdivision_matrix_file = os.path.join(control_mesh_dir,
                                        "subdivision_matrix" + filemod + ".txt")
matrix = (pd.read_table(subdivision_matrix_file,
                                     sep='\s+',
                                     header=None)).values.astype(float)

et_index_file = os.path.join(control_mesh_dir,'ETIndicesSorted' + filemod + '.txt')
if not os.path.exists(et_index_file):
    ValueError('Missing ETIndicesSorted.txt file')
surf_ien = (pd.read_table(et_index_file, sep='\s+',
                                            header=None)).values.astype(int)-1
surf_xyz = np.dot(matrix, control_mesh)

surface_label_file = os.path.join(control_mesh_dir,'surface_region' + filemod + '.txt')
surfs = np.loadtxt(surface_label_file, dtype=int)

io.write_points_cells('check_surf.vtu', surf_xyz, {'triangle': surf_ien}, 
                      cell_data={'Region': [surfs]})

# Surfaces defining BiV
bv_surfs = [0,1,2,3,8,9]
bv_marker = np.isin(surfs, bv_surfs)
surf_ien = surf_ien[bv_marker]

tet = tetgen.TetGen(surf_xyz, surf_ien)
tet.tetrahedralize(order=1, minratio=20, quality=False, verbose=True)
grid = tet.grid
vol_xyz = np.array(grid.points)
vol_ien = grid.cells_dict[10]

# Write surface mesh
# io.write_points_cells('check_surf.vtu', surf_xyz, {'triangle': surf_ien})

# io.write_points_cells('check.vtu', vol_xyz, {'tetra': vol_ien})

# Map volumetric mesh to surface mesh
vol_tree = KDTree(vol_xyz)
surf_tree = KDTree(surf_xyz)
corr = vol_tree.query_ball_tree(surf_tree, r=1e-2)

vol_ien_mod = np.array(corr)[vol_ien]

io.write_points_cells('check.vtu', surf_xyz, {'tetra': vol_ien_mod})
np.save('../volume_template_ien.npy', vol_ien_mod)