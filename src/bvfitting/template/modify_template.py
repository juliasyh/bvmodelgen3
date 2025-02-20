#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/18 15:17:11

@author: Javiera Jilberto Vallejos 
'''


import os

import numpy as np
import pandas as pd
import meshio as io

# Reading surface control mesh
control_mesh_dir = "."
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

io.write_points_cells('template.vtu', surf_xyz, {'triangle': surf_ien}, 
                      cell_data={'Region': [surfs]})

io.write_points_cells('check_control.vtu', control_mesh, {'vertex':np.arange(control_mesh.shape[0])[:,None]})

# Grab PV border
rv_epi_nodes = np.unique(surf_ien[surfs==3])
pv_in_nodes = np.unique(surf_ien[surfs==7])
pv_nodes = np.intersect1d(rv_epi_nodes, pv_in_nodes)
pv_cent_node = np.setdiff1d(pv_in_nodes, pv_nodes)

pv_coords = surf_xyz[pv_cent_node]
pv_mean_coords = np.mean(surf_xyz[pv_nodes], axis=0)