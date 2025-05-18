#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/18 11:50:17

@author: Javiera Jilberto Vallejos 
'''

from PatientData import FittedTemplate
from multiprocessing import Pool
from modelutils import adjust_valve_centroids, calculate_chamber_volumes, calculate_wall_volumes, get_surface_node_normals
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import meshio as io


import pathlib
filepath=pathlib.Path(__file__).parent.resolve()

def fit_frame_template(args):
    frame, img2model_fldr, weight_GP, low_smoothing_weight, transmural_weight, rv_thickness, load_control_points = args
    log = [f'Fitting template for frame {frame}...']

    frame_prefix = f'{img2model_fldr}/frame{frame}_'

    filename = frame_prefix + 'contours.txt'

    template = FittedTemplate(frame_prefix, filemod='_mod2')
    template.load_dataset(filename, weight_GP, load_control_points=load_control_points)

    log += template.fit_template(weight_GP, low_smoothing_weight, transmural_weight, rv_thickness, verbose=False)
    print('\n'.join(log), flush=True)
    surface_mesh = template.bvmodel.get_template_mesh()

    return frame, surface_mesh


def fit_templates_parallel(ncores, img2model_fldr, which_frames, weight_GP, low_smoothing_weight, transmural_weight, rv_thickness, load_control_points=None):
    """
    Fit templates in parallel for the specified frames.

    Parameters:
    - img2model_fldr: Path to the folder containing the images to model.
    - which_frames: List of frames to fit templates for.
    - weight_GP: Weight for Gaussian Process.
    - low_smoothing_weight: Weight for low smoothing.
    - transmural_weight: Weight for transmural.
    - rv_thickness: RV thickness parameter.
    - load_control_points: Control points to load (default is None).

    Returns:
    - List of tuples containing frame number and corresponding surface mesh.
    """
    
    args = [(frame, img2model_fldr, weight_GP, low_smoothing_weight, transmural_weight, rv_thickness, load_control_points) for frame in which_frames]

    surface_meshes = [None] * (max(which_frames) + 1)
    with Pool(ncores) as pool:
        results = pool.map(fit_frame_template, args)

    # Store the results
    for frame, surface_mesh in results:
        surface_meshes[frame] = surface_mesh

    return surface_meshes

def calculate_model_volume(args):
    mesh, label = args
    xyz = mesh.points
    ien = mesh.cells[0].data
    xyz = adjust_valve_centroids(xyz, ien, label)

    # Save modified model
    model = io.Mesh(xyz.copy(), {'triangle': ien}, cell_data={'Region': [label]})

    # Calculate volumes
    lv_vol, rv_vol = calculate_chamber_volumes(xyz, ien, label)
    lv_wall_vol, rv_wall_vol = calculate_wall_volumes(xyz, ien, label)
    return model, lv_vol, rv_vol, lv_wall_vol, rv_wall_vol



def correct_model_chamber_volume(args):
    (model, target_vol, tol, rv_endo_nodes, rv_epi_nodes, rv_distance_field, labels, rv_endo, rv_epi) = args
    _, old_rv_vol = calculate_chamber_volumes(model.points, model.cells[0].data, labels)
    if np.abs(old_rv_vol - target_vol) < tol:
        return model

    # Get normals
    rv_endo_normals = get_surface_node_normals(model.points, model.cells[0].data, labels == rv_endo)
    rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, labels == rv_epi)

    def error_rv_volume(disp_val):
        xyz = model.points.copy()
        xyz[rv_endo_nodes] += disp_val*rv_endo_normals*rv_distance_field[rv_endo_nodes][:,None]
        xyz[rv_epi_nodes] += disp_val*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]
        _, new_rv_vol = calculate_chamber_volumes(xyz, model.cells[0].data, labels)
        return (target_vol - new_rv_vol)**2

    # Find optimal displacement
    res = minimize(error_rv_volume, 0.0, options={'maxiter': 20, 'xrtol': 1e-5, 'disp': False}, tol=1e-5)
    disp_opt = res.x[0]

    model.points[rv_endo_nodes] += disp_opt*rv_endo_normals*rv_distance_field[rv_endo_nodes][:,None]
    model.points[rv_epi_nodes] += disp_opt*rv_epi_normals*rv_distance_field[rv_epi_nodes][:,None]
    return model



def correct_rv_wall_volume(args):
    model, target_vol, tol, rv_epi_nodes, rv_epi, rv_distance_field, labels = args
    _, old_rv_vol = calculate_wall_volumes(model.points, model.cells[0].data, labels)

    if np.abs(old_rv_vol - target_vol) < tol:
        return model

    # Get normals
    rv_epi_normals = get_surface_node_normals(model.points, model.cells[0].data, labels == rv_epi)

    def error_rv_volume(disp):
        xyz = model.points.copy()
        xyz[rv_epi_nodes] += disp * rv_epi_normals * rv_distance_field[rv_epi_nodes][:, None]
        _, new_rv_vol = calculate_wall_volumes(xyz, model.cells[0].data, labels)
        return (target_vol - new_rv_vol) ** 2

    # Find optimal displacement
    res = minimize(error_rv_volume, 0.0, options={'maxiter': 20, 'xrtol': 1e-5, 'disp': False}, tol=1e-5)
    disp = res.x[0]

    model.points[rv_epi_nodes] += disp * rv_epi_normals * rv_distance_field[rv_epi_nodes][:, None]
    return model

def correct_surface_by_volume_parallel(ncores, surfaces, labels):
    """
    Correct the surface meshes by volume in parallel.

    Parameters:
    - ncores: Number of cores to use for parallelization.
    - surface_meshes: List of surface meshes to correct.
    - labels: Labels for the surfaces.

    Returns:
    - List of corrected surface meshes.
    """
    # Load distance field
    rv_distance_field = np.load(f'{filepath}/bvfitting/template/rv_distance_field.npy')

    # Calculate original volumes and initialize models
    lv_volume_og = np.zeros(len(surfaces))
    rv_volume_og = np.zeros(len(surfaces))
    lv_wall_volume_og = np.zeros(len(surfaces))
    rv_wall_volume_og = np.zeros(len(surfaces))

    models = []

    args = [(surfaces[i], labels) for i in range(len(surfaces))]
    with Pool(ncores) as pool:
        results = pool.map(calculate_model_volume, args)

    models = []
    for i, (model, lv_vol, rv_vol, lv_wall_vol, rv_wall_vol) in enumerate(results):
        models.append(model)
        lv_volume_og[i] = lv_vol
        rv_volume_og[i] = rv_vol
        lv_wall_volume_og[i] = lv_wall_vol
        rv_wall_volume_og[i] = rv_wall_vol

    # Set tolerance to 0.1 mm3
    if np.max(rv_volume_og) > 1e4:  # Volume in mm3
        tol = 1e-1
    else:   # Volume in m3 
        tol = 1e-10


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
    parallel_args = []
    for i in range(len(models)):
        model = models[i]
        target_vol = target_rv_volumes[i]
        parallel_args.append((
            model,
            target_vol,
            tol,
            rv_endo_nodes,
            rv_epi_nodes,
            rv_distance_field,
            labels,
            rv_endo,
            rv_epi,
        ))

    with Pool(ncores) as pool:
        models = pool.map(correct_model_chamber_volume, parallel_args)

    
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
    target_vol = rv_wall_volume[0]

    # Prepare arguments for parallel processing of epi correction
    parallel_epi_args = []
    for i in range(len(models)):
        model = models[i]
        parallel_epi_args.append((
            model,
            target_vol,
            tol,
            rv_epi_nodes,
            rv_epi,
            rv_distance_field,
            labels,
        ))

    with Pool(ncores) as pool:
        corrected_meshes = pool.map(correct_rv_wall_volume, parallel_epi_args)

    return corrected_meshes