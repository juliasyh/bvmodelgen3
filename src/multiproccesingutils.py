#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/18 11:50:17

@author: Javiera Jilberto Vallejos 
'''

from PatientData import FittedTemplate
from multiprocessing import Pool


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