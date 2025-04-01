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

def get_surface_normals(points, ien, vol_elems=None):
    points_elems = points[ien]
    if ien.shape[1] == 2:   # Lines
        v1 = points_elems[:,1] - points_elems[:,0]
        v2 = np.array([0,0,1])

        normal = np.cross(v1, v2, axisa=1)
        normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    if ien.shape[1] == 3:

        v1 = points_elems[:,1] - points_elems[:,0]
        v2 = points_elems[:,2] - points_elems[:,0]

        normal = np.cross(v1, v2, axisa=1, axisb=1)
        normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    if vol_elems is None:
        return normal

    elem_midpoint = np.mean(points[vol_elems], axis=1)
    face_midpoint = np.mean(points[ien], axis=1)

    vector = face_midpoint-elem_midpoint
    dot = np.sum(normal*vector, axis=1)
    normal[dot<0] *= -1

    return normal

#%%
template = io.read('src/bvfitting/template/template.vtu')
map_template_to_bv = np.load('src/bvfitting/template/template_to_bv_map.npy')
map_bv_to_template = np.load('src/bvfitting/template/bv_to_template_map.npy')
surfs = template.cell_data['Region'][0]
rv_distance_field = 1-template.point_data['rv_distance_field']

xyz = template.points
ien = template.cells[0].data


lv_volume = []
rv_volume = []
lv_wall_volume = []
rv_wall_volume = []
models = []
for i in range(30):
    bv_surface = io.read(f'test_data/Images/img2model/frame{i}_bv_surface_smoothed.stl')
    template.points[map_template_to_bv] = bv_surface.points

    # Modify valve centroids
    xyz = adjust_valve_centroids(xyz, ien, surfs)
    template.points = xyz

    model = io.Mesh(xyz.copy(), {'triangle': ien}, cell_data={'Region': [surfs]})
    models.append(model)

    # Calculate volumes
    lv_vol, rv_vol = calculate_chamber_volumes(xyz, ien, surfs)
    lv_wall_vol, rv_wall_vol = calculate_wall_volumes(xyz, ien, surfs)

    lv_volume.append(lv_vol)
    rv_volume.append(rv_vol)
    lv_wall_volume.append(lv_wall_vol)
    rv_wall_volume.append(rv_wall_vol)
    
#%%
from scipy.interpolate import interp1d
lv_volume = np.array(lv_volume)
rv_volume = np.array(rv_volume)
lv_sv = np.max(lv_volume) - np.min(lv_volume)
rv_sv = np.max(rv_volume) - np.min(rv_volume)
rv_ed_correct = np.min(rv_volume) + lv_sv
rv_ed = np.max(rv_volume)
mult = rv_ed_correct/rv_ed

lowest_vol_idx = np.argmin(rv_volume)
scaling_func = interp1d([0, lowest_vol_idx, len(rv_volume)-1], [mult, 1, mult])
scaling = scaling_func(np.arange(len(rv_volume)))
target_rv_volumes = rv_volume*scaling

#%%

def cell_data_to_point_data(array, ien, xyz, method='mean'):
    node_value = np.zeros((len(xyz), array.shape[1]))
    node_cont = np.zeros(len(xyz))
    for e in range(ien.shape[0]):
        nodes = ien[e]
        for i, node in enumerate(nodes):
            node_value[node] += array[e]
            node_cont[node] += 1
    
    if method == 'mean':
        node_value[node_cont > 0] /= node_cont[node_cont > 0,None]

    node_value = node_value[node_cont > 0]

    return node_value

def get_surface_node_normals(xyz, ien, patch):
    patch_elems = ien[patch]
    patch_nodes = np.unique(patch_elems)
    patch_normals = get_surface_normals(xyz, patch_elems)
    patch_node_normals = cell_data_to_point_data(patch_normals, patch_elems, xyz)
    patch_centroid = np.mean(xyz[patch_nodes], axis=0)
    dot = np.sum(patch_node_normals*(xyz[patch_nodes]-patch_centroid), axis=1)
    patch_node_normals[dot<0] *= -1
    return patch_node_normals

model = models[-1]
rv_endo = 2
rv_epi = 3

rv_endo_nodes = np.unique(model.cells[0].data[surfs == rv_endo])
rv_epi_nodes = np.unique(model.cells[0].data[surfs == rv_epi])
rv_endo_normals = get_surface_node_normals(model.points, model.cells[0].data, surfs == rv_endo)
rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, surfs == rv_epi)


#%% Move endo
from scipy.optimize import minimize

disp = 0.0
for i in range(len(models)):
    print(i)
    model = models[i]

    target_vol = target_rv_volumes[i]
    old_lv_vol, old_rv_vol = calculate_chamber_volumes(model.points, model.cells[0].data, surfs)

    if np.abs(old_rv_vol - target_vol) < 1e-5:
        continue

    # Get normals
    rv_endo_normals = get_surface_node_normals(model.points, model.cells[0].data, surfs == rv_endo)
    rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, surfs == rv_epi)

    xyz = model.points.copy()

    def error_rv_volume(disp):
        xyz = model.points.copy()
        xyz[rv_endo_nodes] += disp*rv_endo_normals*rv_distance_field[rv_endo_nodes][:,None]
        xyz[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]
        _, new_rv_vol = calculate_chamber_volumes(xyz, model.cells[0].data, surfs)
        return (target_vol - new_rv_vol)**2

    # Find optimal displacement
    res = minimize(error_rv_volume, disp, options={'maxiter': 20, 'xrtol': 1e-5, 'disp': True}, tol=1e-5)
    disp = res.x[0]

    model.points[rv_endo_nodes] += disp*rv_endo_normals*rv_distance_field[rv_endo_nodes][:,None]
    model.points[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]

#%% Check volumes
new_lv_volume = []
new_rv_volume = []
new_lv_wall_volume = []
new_rv_wall_volume = []
for i in range(len(models)):
    model = models[i]
    xyz = model.points

    # Calculate volumes
    lv_vol, rv_vol = calculate_chamber_volumes(xyz, ien, surfs)
    lv_wall_vol, rv_wall_vol = calculate_wall_volumes(xyz, ien, surfs)

    new_lv_volume.append(lv_vol)
    new_rv_volume.append(rv_vol)
    new_lv_wall_volume.append(lv_wall_vol)
    new_rv_wall_volume.append(rv_wall_vol)

#%% Move epi to get consistent wall volumes

disp = 0.0
target_vol = new_rv_wall_volume[0]
for i in range(len(models)):
    print(i)
    model = models[i]

    old_lv_vol, old_rv_vol = calculate_wall_volumes(model.points, model.cells[0].data, surfs)

    if np.abs(old_rv_vol - target_vol) < 1e-5:
        continue

    # Get normals
    rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, surfs == rv_epi)

    xyz = model.points.copy()

    def error_rv_volume(disp):
        xyz = model.points.copy()
        xyz[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]
        _, new_rv_vol = calculate_wall_volumes(xyz, model.cells[0].data, surfs)
        return (target_vol - new_rv_vol)**2

    # Find optimal displacement
    res = minimize(error_rv_volume, disp, options={'maxiter': 20, 'xrtol': 1e-5, 'disp': True}, tol=1e-5)
    disp = res.x[0]

    model.points[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]

#%% Save new models
for i, model in enumerate(models):
    io.write(f'test_data/Images/img2model/frame{i}_bv_surface_smoothed_mod.stl', model)



#%% Check volumes
new_lv_volume = []
new_rv_volume = []
new_lv_wall_volume = []
new_rv_wall_volume = []
for i in range(30):
    model = models[i]
    xyz = model.points

    # Calculate volumes
    lv_vol, rv_vol = calculate_chamber_volumes(xyz, ien, surfs)
    lv_wall_vol, rv_wall_vol = calculate_wall_volumes(xyz, ien, surfs)

    new_lv_volume.append(lv_vol)
    new_rv_volume.append(rv_vol)
    new_lv_wall_volume.append(lv_wall_vol)
    new_rv_wall_volume.append(rv_wall_vol)


#%%
plt.figure(1, clear=True)
plt.plot(np.array(lv_volume)/1000, color='r', marker='.', ls='--')
plt.plot(np.array(rv_volume)/1000, color='b', marker='.', ls='--')
plt.plot(np.array(new_lv_volume)/1000, color='r', marker='.')
plt.plot(np.array(new_rv_volume)/1000, color='b', marker='.')
plt.plot([], [], color='r', label='LV')
plt.plot([], [], color='b', label='RV')
plt.plot([], [], color='k', linestyle='-', label='No smoothing')
plt.plot([], [], color='k', linestyle='--', label='Smooth')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Volume (ml)')
plt.title('Chamber volumes')
plt.savefig('mod_chamber_volumes.png')

#%%
plt.figure(1, clear=True)
plt.plot(np.array(lv_wall_volume)/1000, color='r', marker='.', ls='--')
plt.plot(np.array(rv_wall_volume)/1000, color='b', marker='.', ls='--')
plt.plot(np.array(new_lv_wall_volume)/1000, color='r', marker='.')
plt.plot(np.array(new_rv_wall_volume)/1000, color='b', marker='.')
plt.plot([], [], color='r', label='LV')
plt.plot([], [], color='b', label='RV')
plt.plot([], [], color='k', linestyle='-', label='No smoothing')
plt.plot([], [], color='k', linestyle='--', label='Smooth')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Volume (ml)')
plt.title('Wall volumes')
plt.savefig('mod_wall_volumes.png')

