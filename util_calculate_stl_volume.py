#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/19 10:20:44

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
import numpy as np
import meshio as io
from scipy.spatial import cKDTree
import trimesh


def adjust_valve_centroids(points, cells, surfs):
    rv_epi = 3
    mv = 4
    av = 5
    tv = 6
    pv = 7
    lv_epi = 8

    pv_cent_node = 5729
    av_cent_node = 5655
    mv_cent_node = 5630
    tv_cent_node = 5696


    # Grab relevant nodes
    lv_epi_nodes = np.unique(cells[surfs == lv_epi])
    rv_epi_nodes = np.unique(cells[surfs == rv_epi])

    mv_nodes = np.intersect1d(lv_epi_nodes, np.unique(cells[surfs == mv]))
    av_nodes = np.intersect1d(lv_epi_nodes, np.unique(cells[surfs == av]))
    tv_nodes = np.intersect1d(rv_epi_nodes, np.unique(cells[surfs == tv]))
    pv_nodes = np.intersect1d(rv_epi_nodes, np.unique(cells[surfs == pv]))

    points[mv_cent_node] = np.mean(points[mv_nodes], axis=0)
    points[av_cent_node] = np.mean(points[av_nodes], axis=0)
    points[tv_cent_node] = np.mean(points[tv_nodes], axis=0)
    points[pv_cent_node] = np.mean(points[pv_nodes], axis=0)
    return points



def get_chamber_meshes(points, cells, surfs):
    # LV surface
    lv_surfs = [0,4,5]
    lv_marker = np.isin(surfs, lv_surfs)
    lv_mesh = io.Mesh(points, {'triangle': cells[lv_marker]})

    # RV surface
    rv_surfs = [1,2,6,7]
    rv_marker = np.isin(surfs, rv_surfs)
    rv_mesh = io.Mesh(points, {'triangle': cells[rv_marker]})

    return lv_mesh, rv_mesh

def get_lv_rv_surface_mesh(points, cells, surfs):
    # Surfaces defining BiV
    lv_surfs = [0,1,8,9]
    rv_surfs = [2,3,9]

    lv_marker = np.isin(surfs, lv_surfs)
    lv_mesh = io.Mesh(points, {'triangle': cells[lv_marker]})

    rv_marker = np.isin(surfs, rv_surfs)
    rv_mesh = io.Mesh(points, {'triangle': cells[rv_marker]})

    return lv_mesh, rv_mesh


def get_enclosed_volume(xyz, faces):
    mesh = trimesh.Trimesh(xyz, faces)
    trimesh.repair.fix_normals(mesh)
    return mesh.volume


def calculate_chamber_volumes(points, cells, surfs):
    lv_mesh, rv_mesh = get_chamber_meshes(points, cells, surfs)
    io.write('check_rv.vtu', rv_mesh)
    io.write('check_lv.vtu', lv_mesh)

    return get_enclosed_volume(lv_mesh.points, lv_mesh.cells[0].data), \
            get_enclosed_volume(rv_mesh.points, rv_mesh.cells[0].data)

def calculate_wall_volumes(points, cells, surfs):
    lv_mesh, rv_mesh = get_lv_rv_surface_mesh(points, cells, surfs)

    return get_enclosed_volume(lv_mesh.points, lv_mesh.cells[0].data), \
            get_enclosed_volume(rv_mesh.points, rv_mesh.cells[0].data)


template = io.read('src/bvfitting/template/template.vtu')
mapp = np.load('src/bvfitting/template/template_to_bv_map.npy')
surfs = template.cell_data['Region'][0]

xyz = template.points
ien = template.cells[0].data

lv_volume = []
rv_volume = []
lv_wall_volume = []
rv_wall_volume = []
for i in range(30):
    bv_surface = io.read(f'test_data/Images/img2model/frame{i}_bv_surface.stl')
    template.points[mapp] = bv_surface.points

    # Modify valve centroids
    xyz = adjust_valve_centroids(xyz, ien, surfs)
    template.points = xyz

    # Calculate volumes
    lv_vol, rv_vol = calculate_chamber_volumes(xyz, ien, surfs)
    lv_wall_vol, rv_wall_vol = calculate_wall_volumes(xyz, ien, surfs)

    lv_volume.append(lv_vol)
    rv_volume.append(rv_vol)
    lv_wall_volume.append(lv_wall_vol)
    rv_wall_volume.append(rv_wall_vol)


lv_volume_smooth = []
rv_volume_smooth = []
lv_wall_volume_smooth = []
rv_wall_volume_smooth = []
for i in range(30):
    bv_surface = io.read(f'test_data/Images/img2model/frame{i}_bv_surface_smoothed.stl')
    template.points[mapp] = bv_surface.points

    # Modify valve centroids
    xyz = adjust_valve_centroids(xyz, ien, surfs)
    template.points = xyz

    # Calculate volumes
    lv_vol, rv_vol = calculate_chamber_volumes(xyz, ien, surfs)
    lv_wall_vol, rv_wall_vol = calculate_wall_volumes(xyz, ien, surfs)

    lv_volume_smooth.append(lv_vol)
    rv_volume_smooth.append(rv_vol)
    lv_wall_volume_smooth.append(lv_wall_vol)
    rv_wall_volume_smooth.append(rv_wall_vol)

#%%
plt.figure(1, clear=True)
plt.plot(np.array(lv_volume)/1000, color='r', marker='.')
plt.plot(np.array(lv_volume_smooth)/1000, color='r', linestyle='--', marker='.')
plt.plot(np.array(rv_volume)/1000, color='b', marker='.')
plt.plot(np.array(rv_volume_smooth)/1000, color='b', linestyle='--', marker='.')
plt.plot([], [], color='r', label='LV')
plt.plot([], [], color='b', label='RV')
plt.plot([], [], color='k', linestyle='-', label='No smoothing')
plt.plot([], [], color='k', linestyle='--', label='Smooth')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Volume (ml)')
plt.title('Chamber volumes')
plt.savefig('chamber_volumes.png')

#%% 
plt.figure(1, clear=True)
plt.plot(np.array(lv_wall_volume)/1000, color='r', marker='.')
plt.plot(np.array(lv_wall_volume_smooth)/1000, color='r', linestyle='--', marker='.')
plt.plot(np.array(rv_wall_volume)/1000, color='b', marker='.')
plt.plot(np.array(rv_wall_volume_smooth)/1000, color='b', linestyle='--', marker='.')
plt.plot([], [], color='r', label='LV')
plt.plot([], [], color='b', label='RV')
plt.plot([], [], color='k', linestyle='-', label='No smoothing')
plt.plot([], [], color='k', linestyle='--', label='Smooth')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Volume (ml)')
plt.title('Wall volumes')
plt.savefig('wall_volumes.png')