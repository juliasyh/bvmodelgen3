#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:26:30 2023

@author: Javiera Jilberto Vallejos
"""

from PatientData import PatientData
from time import time

if __name__ ==  '__main__':

    ##############################################################
    ######  USER INPUTS AND OPTIONS ##############################
    ##############################################################

    # Options
    interpolate_segmentations = 'max'   # ('max', 'min', int, None) means to interpolate the segmentations to the max or min number of frames, 
                                        # or to a specific number of frames. None means not to interpolate.
    nn_segmentation = False             # Use nn to create segmentations, if False, it will load them from the paths defined in segs
    align_segmentations = True
    visualize = False
    ncores = 5                          # Number of cores to use for parallelization.
    smooth_in_time = True
    correct_using_volumes = True
    load_contours = True
    load_surfaces = None                # Load surfaces. None means not to load. 'initial' means after the initial fitting, 
                                        # 'smooth', after smoothing, 'corrected' after volume correction.
    
    # Inputs
    output_path = 'Images/'   # where all the output will be saved
    imgs_path = 'Images/'    # Dummy variable to define common paths, not truly needed if you define the paths directly

    imgs = {'sa': imgs_path + 'SA',
            'la_2ch': imgs_path + 'LA_2CH',
            'la_3ch': imgs_path + 'LA_3CH',
            'la_4ch': imgs_path + 'LA_4CH'}

    # Which frames to process
    which_frames = None  # None means all frames. Remember Python starts with 0!

    # Paths to the valve segmentations    
    valves_3ch_slice = []  # The slice to use for the 3-chamber view. Only use one slice!
    valves = {'la_2ch': imgs_path + 'LA_2CH_valves',
              'la_3ch': imgs_path + 'LA_3CH_valves',
              'la_4ch': imgs_path + 'LA_4CH_valves'}

    # Paths to segmentations. Note that if nn_segmentation is True, these paths are not used.    
    segs = {'sa': imgs_path + 'sa_seg',
            'la_2ch': imgs_path + 'la_2ch_seg',
            'la_3ch': imgs_path + 'la_3ch_seg',
            'la_4ch': imgs_path + 'la_4ch_seg'}
    

    global_time = time()
    
    ##############################################################
    ######  IMAGE INTERPOLATION #################################
    ##############################################################

    interp_time = time()
    if (interpolate_segmentations == 'max' 
        or interpolate_segmentations == 'min' 
        or isinstance(interpolate_segmentations, int)):
        from imgutils import interpolate_scans
        interp_views = interpolate_scans(imgs, interpolate_segmentations)
        # Update the images with the interpolated ones
        for view in interp_views:
            imgs[view] = imgs[view] + '_interp'     
    interp_time = time() - interp_time


    ##############################################################
    ######  TEMPLATE FITTING AND SURFACE GENERATION ##############
    ##############################################################

    # Initialize the PatientData object
    pdata = PatientData(imgs, output_path)

    # Segment the images
    if nn_segmentation:
        nn_time = time()
        segs = pdata.unet_generate_segmentations(segs)
        nn_time = time() - nn_time

    # Load segmentations
    pdata.load_segmentations(segs)
    pdata.cmr_data.clean_segmentations()

    # Generate surfaces
    if load_surfaces is None:
        contours_time = time()
        pdata.cmr_data.extract_contours(visualize=visualize, align=align_segmentations, which_frames=which_frames)
        pdata.find_valves(valves, slices_3ch=valves_3ch_slice, visualize=visualize)
        pdata.generate_contours(which_frames=which_frames, visualize=visualize)
        contours_time = time() - contours_time

        fit_time = time()
        surfaces = pdata.fit_templates(which_frames=which_frames, parallelize=ncores)
        pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames, prefix='step0_')
        fit_time = time() - fit_time

        if smooth_in_time:
            correct_time = time()
            surfaces = pdata.smooth_surfaces_in_time(surfaces, which_frames=which_frames)
            pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames, prefix='step1_')
            if correct_using_volumes:
                surfaces = pdata.correct_surfaces_by_volumes(surfaces, which_frames=which_frames, parallelize=ncores)
                pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames, prefix='step2_')
            correct_time = time() - correct_time

    elif load_surfaces == 'initial':
        surfaces = pdata.load_surfaces(which_frames=which_frames, prefix='step0_')
        if smooth_in_time:
            surfaces = pdata.smooth_surfaces_in_time(surfaces)
            pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames, prefix='step1_')
            if correct_using_volumes:
                surfaces = pdata.correct_surfaces_by_volumes(surfaces, which_frames=which_frames, prefix='step2_')
                pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames)

    elif load_surfaces == 'smooth':
        surfaces = pdata.load_surfaces(which_frames=which_frames, prefix='step1_')
        if correct_using_volumes:
            surfaces = pdata.correct_surfaces_by_volumes(surfaces, which_frames=which_frames)
            pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames, prefix='step2_')

    elif load_surfaces == 'corrected':
        surfaces = pdata.load_surfaces(which_frames=which_frames, prefix='step2_')


    # ##############################################################
    # ######  VOLUME CALCULATION  ##################################
    # ##############################################################
    vol_time = time()
    lv_volume, rv_volume = pdata.calculate_chamber_volumes(surfaces, which_frames=which_frames)
    lv_wall_volume, rv_wall_volume = pdata.calculate_wall_volumes(surfaces, which_frames=which_frames)
    vol_time = time() - vol_time


    # ##############################################################
    # ######  TIME LOG  ############################################
    # ##############################################################
    global_time = time() - global_time

    if (interpolate_segmentations == 'max' 
        or interpolate_segmentations == 'min' 
        or isinstance(interpolate_segmentations, int)):
        print('Time taken for interpolation: ', interp_time)
    if nn_segmentation:
        print('Time taken for nn segmentation: ', nn_time)
    print('Time taken for contour extraction: ', contours_time)
    print('Time taken for template fitting: ', fit_time)
    print('Time taken for surface correction: ', correct_time)
    print('Time taken for volume calculation: ', vol_time)
    print('Total time taken: ', global_time)