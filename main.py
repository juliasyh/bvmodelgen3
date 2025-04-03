#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:26:30 2023

@author: Javiera Jilberto Vallejos
"""

from PatientData import PatientData

if __name__ ==  '__main__':

    ##############################################################
    ######  USER INPUTS AND OPTIONS ##############################
    ##############################################################

    # Options
    nn_segmentation = True     # Use nn to create segmentations, if False, it will load them from the paths defined in segs
    align_segmentations = True
    visualize = False
    smooth_in_time = True
    correct_using_volumes = True
    load_surfaces = False       # Load surfaces from output folder instead of generating them

    # Inputs
    output_path = 'test_data/'   # where all the output will be saved
    imgs_path = 'test_data/'     # Dummy variable to define common paths, not truly needed if you define the paths directly

    imgs = {'sa': imgs_path + 'SA',
            'la_2ch': imgs_path + 'LA_2CH',
            'la_3ch': imgs_path + 'LA_3CH',
            'la_4ch': imgs_path + 'LA_4CH'}

    # Which frames to process
    which_frames = None  # None means all frames

    # Paths to the valve segmentations    
    valves_3ch_slice = [1]  # The slice to use for the 3-chamber view. Only use one slice!
    valves = {'la_2ch': imgs_path + 'LA_2CH_valves', 
              'la_3ch': imgs_path + 'LA_3CH_valves',    
              'la_4ch': imgs_path + 'LA_4CH_valves'}

    # Paths to segmentations. Note that if nn_segmentation is True, these paths are not used.    
    segs = {'sa': imgs_path + 'sa_seg',
            'la_2ch': imgs_path + 'la_2ch_seg',
            'la_3ch': imgs_path + 'la_3ch_seg',
            'la_4ch': imgs_path + 'la_4ch_seg'}
    

    ##############################################################
    ######  TEMPLATE FITTING AND SURFACE GENERATION ##############
    ##############################################################

    # Initialize the PatientData object
    pdata = PatientData(imgs, output_path)

    # Segment the images
    if nn_segmentation:
        segs = pdata.unet_generate_segmentations(segs)

    # Load segmentations
    pdata.load_segmentations(segs)

    # Generate surfaces
    if load_surfaces:
        surfaces = pdata.load_surfaces(which_frames=which_frames)
    else:
        pdata.cmr_data.extract_contours(visualize=visualize, align=align_segmentations, which_frames=which_frames)
        pdata.find_valves(valves, slices_3ch=valves_3ch_slice, visualize=visualize)
        pdata.generate_contours(which_frames=which_frames, visualize=visualize)

        # surfaces = pdata.load_surfaces()
        if smooth_in_time:
            surfaces = pdata.fit_templates(which_frames=which_frames)
            pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames)
            surfaces = pdata.smooth_surfaces_in_time(surfaces)
            pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames)
            if correct_using_volumes:
                surfaces = pdata.correct_surfaces_by_volumes(surfaces, which_frames=which_frames)
                pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames)
        else:
            # No postprocess, meshes are saved after fitting
            surfaces = pdata.fit_templates(which_frames=which_frames)
        pdata.save_bv_surfaces(surfaces, mesh_subdivisions=2, which_frames=which_frames)

    # Calculate volumes
    lv_volume, rv_volume = pdata.calculate_chamber_volumes(surfaces, which_frames=which_frames)
    lv_wall_volume, rv_wall_volume = pdata.calculate_wall_volumes(surfaces, which_frames=which_frames)