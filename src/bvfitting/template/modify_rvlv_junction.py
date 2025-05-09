#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/18 15:17:11

@author: Javiera Jilberto Vallejos 
'''


import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import meshio as io
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D


def get_normal_plane_svd(points):   # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid

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

valve_label_file = os.path.join(control_mesh_dir,'valve_elems' + filemod + '.txt')
valve_elems = np.loadtxt(valve_label_file, dtype = int)

rv_sep_nodes = np.unique(surf_ien[surfs==1])
rv_epi_nodes = np.unique(surf_ien[surfs==3])
rvlv_nodes = np.unique(surf_ien[surfs==9])
mv_nodes = np.unique(surf_ien[surfs==4])
endo_nodes = np.intersect1d(rv_sep_nodes, rvlv_nodes)


add_epi_elems = np.loadtxt('add_epi.csv', usecols=1, dtype=int, skiprows=1, delimiter=',')
add_epi_nodes = np.unique(surf_ien[add_epi_elems])
epi_nodes = np.loadtxt('new_rvlv_nodes.csv', usecols=0, dtype=int, skiprows=1, delimiter=',')
epi_nodes = epi_nodes[:-1]

nodes = np.union1d(endo_nodes, epi_nodes)

surfs[add_epi_elems] = 8

# 3D scatter plot for endo and epi nodes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot endo nodes
# endo_coords = surf_xyz[endo_nodes]
# ax.scatter(endo_coords[:, 0], endo_coords[:, 1], endo_coords[:, 2], c='blue', label='Endo Nodes', s=10)

# # Plot epi nodes
# epi_coords = surf_xyz[epi_nodes]
# ax.scatter(epi_coords[:, 0], epi_coords[:, 1], epi_coords[:, 2], c='red', label='Epi Nodes', s=10)

# # Labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.title('3D Scatter Plot of Endo and Epi Nodes')
# plt.show()

normal, centroid = get_normal_plane_svd(surf_xyz[nodes])
normal = normal/np.linalg.norm(normal)
la_normal, _ = get_normal_plane_svd(surf_xyz[mv_nodes])
la_normal = la_normal/np.linalg.norm(la_normal)

vec3 = normal
vec2 = np.cross(vec3, la_normal)
vec2 = vec2/np.linalg.norm(vec2)
vec1 = np.cross(vec2, vec3)
vec1 = vec1/np.linalg.norm(vec1)

# Rotation matrix
R = np.array([vec1, vec2, vec3])

projected_nodes = np.dot(surf_xyz[nodes] - centroid, R.T)
projected_nodes = projected_nodes[:, :2]
projected_nodes = np.vstack((projected_nodes, np.zeros((1, 2))))

# Triangulate
tri = Delaunay(projected_nodes[:, :2])
triangles = tri.simplices
midpoints = np.mean(projected_nodes[triangles], axis=1)

# Create a polygon from the endo_nodes
endo_proj_nodes = projected_nodes[:len(endo_nodes)]
angle = np.arctan2(endo_proj_nodes[:, 1], endo_proj_nodes[:, 0])
sorted_indices = np.argsort(angle)
endo_proj_nodes = endo_proj_nodes[sorted_indices]
epi_polygon = Polygon(endo_proj_nodes)

# Create a polygon from the epi_nodes
epi_proj_nodes = projected_nodes[len(endo_nodes):len(endo_nodes) + len(epi_nodes)]
angle = np.arctan2(epi_proj_nodes[:, 1], epi_proj_nodes[:, 0])
sorted_indices = np.argsort(angle)
epi_proj_nodes = epi_proj_nodes[sorted_indices]
endo_polygon = Polygon(epi_proj_nodes)

#%%
# epi_proj_nodes = projected_nodes[len(endo_nodes):len(endo_nodes) + len(epi_nodes)]
# # plt.triplot(projected_nodes[:, 0], projected_nodes[:, 1], triangles, color='gray')
# # plt.plot(epi_proj_nodes[:,0], epi_proj_nodes[:,1], '.')
# # plt.plot(epi_proj_nodes[:79,0], epi_proj_nodes[:79,1], '.')

#%%
# Filter triangles to remove those inside the polygon
valid_triangles = []
for i in range(len(triangles)):
    triangle_midpoint = Point(midpoints[i])
    # print(endo_polygon.contains(triangle_midpoint))
    if not endo_polygon.contains(triangle_midpoint):
        valid_triangles.append(i)

valid_triangles = np.array(valid_triangles)

triangles = triangles[valid_triangles]
midpoints = np.mean(projected_nodes[triangles], axis=1)

# Filter triangles to remove those outside the polygon
valid_triangles = []
for i in range(len(triangles)):
    triangle_midpoint = Point(midpoints[i])
    # print(endo_polygon.contains(triangle_midpoint))
    if epi_polygon.contains(triangle_midpoint):
        valid_triangles.append(i)

valid_triangles = np.array(valid_triangles)
triangles = triangles[valid_triangles]

# Create new surface
new_connectivity = nodes[triangles]

surf_ien = np.vstack((surf_ien, new_connectivity))
surfs = np.hstack((surfs, np.zeros(new_connectivity.shape[0]) + 14))

io.write_points_cells('check.vtu', surf_xyz, {'triangle': surf_ien}, 
                      cell_data={'Region': [surfs]})

#%% Save new mesh
filemod='_mod2'
et_index_file_new = os.path.join(control_mesh_dir,'ETIndicesSorted' + filemod + '.txt')
np.savetxt(et_index_file_new, surf_ien+1, fmt='%d', delimiter=' ')

surface_label_file_new = os.path.join(control_mesh_dir,'surface_region' + filemod + '.txt')
np.savetxt(surface_label_file_new, surfs, fmt='%d', delimiter=' ')

model_file_new = os.path.join(control_mesh_dir,"model" + filemod + ".txt")
np.savetxt(model_file_new, control_mesh, fmt='%f', delimiter=' ')

subdivision_matrix_file_new = os.path.join(control_mesh_dir,
                                        "subdivision_matrix" + filemod + ".txt")
np.savetxt(subdivision_matrix_file_new, matrix, fmt='%f', delimiter=' ')

valve_label_file_new = os.path.join(control_mesh_dir,'valve_elems' + filemod + '.txt')
np.savetxt(valve_label_file_new, valve_elems, fmt='%d', delimiter=' ')



#%%
plt.triplot(projected_nodes[:, 0], projected_nodes[:, 1], triangles, color='gray')
plt.plot(projected_nodes[:,0], projected_nodes[:,1], '.')
# plt.plot(midpoints[:,0], midpoints[:,1], 'o', color='red')
