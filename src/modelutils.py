#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/01 14:12:16

@author: Javiera Jilberto Vallejos 
'''


import numpy as np

import meshio as io
import trimesh

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from tqdm import tqdm


def subdivide_mesh(mesh, n):
    """
    Subdivide a mesh using trimesh

    Parameters:
    mesh: meshio triangular mesh
    n: number of subdivisions
    """

    # Generate trimesh object
    tri = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells[0].data)

    # Subdivide the mesh
    for i in range(n):
        tri = tri.subdivide()

    # Convert back to meshio object
    new_mesh = io.Mesh(points=tri.vertices, cells=[("triangle", tri.faces)])
    return new_mesh


def extract_subregion(mesh, labels, regions):
    """
    Extract a subregion from a mesh based on labels

    Parameters:
    mesh: meshio triangular mesh
    labels: numpy array of labels
    regions: list of labels to extract

    Returns:
    submesh: meshio triangular mesh with the extracted subregion
    """

    # Get the indices of the elements that belong to the specified regions
    region_indices = np.isin(labels, regions)

    # Create a new mesh with only the specified regions
    submesh = io.Mesh(points=mesh.points, cells=[("triangle", mesh.cells[0].data[region_indices])])
    
    return submesh


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


def volume_match_correction(surfaces, labels):
    # Load distance field
    rv_distance_field = np.load('src/bvfitting/template/rv_distance_field.npy')

    # Calculate original volumes and initialize models
    lv_volume_og = np.zeros(len(surfaces))
    rv_volume_og = np.zeros(len(surfaces))
    lv_wall_volume_og = np.zeros(len(surfaces))
    rv_wall_volume_og = np.zeros(len(surfaces))

    models = []
    for i in range(len(surfaces)):
        mesh = surfaces[i]
        xyz = mesh.points
        ien = mesh.cells[0].data
        xyz = adjust_valve_centroids(xyz, ien, labels)

        # Save modified model
        model = io.Mesh(xyz.copy(), {'triangle': ien}, cell_data={'Region': [labels]})
        models.append(model)
            
        # Calculate volumes
        lv_volume_og[i], rv_volume_og[i] = calculate_chamber_volumes(xyz, ien, labels)
        lv_wall_volume_og[i], rv_wall_volume_og[i] = calculate_wall_volumes(xyz, ien, labels)


    # Correct the traces
    lv_volume = lv_volume_og.copy()
    rv_volume = rv_volume_og.copy()
    lv_sv = np.max(lv_volume) - np.min(lv_volume)
    rv_sv = np.max(rv_volume) - np.min(rv_volume)
    rv_ed_correct = np.min(rv_volume) + lv_sv
    rv_ed = np.max(rv_volume)
    mult = rv_ed_correct/rv_ed

    lowest_vol_idx = np.argmin(rv_volume)
    scaling_func = interp1d([0, lowest_vol_idx, len(rv_volume)-1], [mult, 1, mult])
    scaling = scaling_func(np.arange(len(rv_volume)))
    target_rv_volumes = rv_volume*scaling

    # Grabbing rv nodes
    model = models[-1]
    rv_endo = 2
    rv_epi = 3

    rv_endo_nodes = np.unique(model.cells[0].data[labels == rv_endo])
    rv_epi_nodes = np.unique(model.cells[0].data[labels == rv_epi])

    # Modify walls to match chamber volumes
    disp = 0.0
    for i in range(len(models)):
        print(i)
        model = models[i]

        target_vol = target_rv_volumes[i]
        old_lv_vol, old_rv_vol = calculate_chamber_volumes(model.points, model.cells[0].data, labels)

        if np.abs(old_rv_vol - target_vol) < 1e-5:
            continue

        # Get normals
        rv_endo_normals = get_surface_node_normals(model.points, model.cells[0].data, labels == rv_endo)
        rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, labels == rv_epi)

        xyz = model.points.copy()

        def error_rv_volume(disp):
            xyz = model.points.copy()
            xyz[rv_endo_nodes] += disp*rv_endo_normals*rv_distance_field[rv_endo_nodes][:,None]
            xyz[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]
            _, new_rv_vol = calculate_chamber_volumes(xyz, model.cells[0].data, labels)
            return (target_vol - new_rv_vol)**2

        # Find optimal displacement
        res = minimize(error_rv_volume, disp, options={'maxiter': 20, 'xrtol': 1e-5, 'disp': True}, tol=1e-5)
        disp = res.x[0]

        model.points[rv_endo_nodes] += disp*rv_endo_normals*rv_distance_field[rv_endo_nodes][:,None]
        model.points[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]


    # Calculate new volumes
    lv_volume = np.zeros(len(models))
    rv_volume = np.zeros(len(models))
    lv_wall_volume = np.zeros(len(models))
    rv_wall_volume = np.zeros(len(models))
    for i in range(len(models)):
        model = models[i]
        lv_volume[i], rv_volume[i] = calculate_chamber_volumes(model.points, model.cells[0].data, labels)
        lv_wall_volume[i], rv_wall_volume[i] = calculate_wall_volumes(model.points, model.cells[0].data, labels)


    # Move epi to match initial wall volume
    disp = 0.0
    target_vol = rv_wall_volume[0]
    for i in range(len(models)):
        print(i)
        model = models[i]

        _, old_rv_vol = calculate_wall_volumes(model.points, model.cells[0].data, labels)

        if np.abs(old_rv_vol - target_vol) < 1e-5:
            continue

        # Get normals
        rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, labels == rv_epi)

        xyz = model.points.copy()

        def error_rv_volume(disp):
            xyz = model.points.copy()
            xyz[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]
            _, new_rv_vol = calculate_wall_volumes(xyz, model.cells[0].data, labels)
            return (target_vol - new_rv_vol)**2

        # Find optimal displacement
        res = minimize(error_rv_volume, disp, options={'maxiter': 20, 'xrtol': 1e-5, 'disp': True}, tol=1e-5)
        disp = res.x[0]

        model.points[rv_epi_nodes] += disp*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]

    return models


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
    lv_surfs = [0,1,8,9,10,12]
    rv_surfs = [2,3,9,11,13]

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