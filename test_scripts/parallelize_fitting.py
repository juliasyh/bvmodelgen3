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
    interpolate_segmentations = 'max'  # ('max', 'min', int, None) means to interpolate the segmentations to the max or min number of frames, 
                                       # or to a specific number of frames. None means not to interpolate.
    nn_segmentation = False             # Use nn to create segmentations, if False, it will load them from the paths defined in segs
    align_segmentations = True
    visualize = False
    smooth_in_time = True
    correct_using_volumes = True
    load_contours = True
    load_surfaces = None                # Load surfaces. None means not to load. 'initial' means after the initial fitting, 
                                        # 'smooth', after smoothing, 'corrected' after volume correction.
    
    # Inputs
    output_path = '../test_data/Images/'   # where all the output will be saved
    imgs_path = '../test_data/Images/'    # Dummy variable to define common paths, not truly needed if you define the paths directly

    imgs = {'sa': imgs_path + 'SA',
            'la_2ch': imgs_path + 'LA_2CH',
            'la_3ch': imgs_path + 'LA_3CH',
            'la_4ch': imgs_path + 'LA_4CH'}

    # Which frames to process
    which_frames = [0,1]  # None means all frames. Remember Python starts with 0!

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
    

    global_time = time()
    
    ##############################################################
    ######  TEMPLATE FITTING AND SURFACE GENERATION ##############
    ##############################################################

    # Initialize the PatientData object
    pdata = PatientData(imgs, output_path)

    # Load segmentations
    pdata.load_segmentations(segs)
    pdata.cmr_data.clean_segmentations()

    # Generate surfaces
    contours_time = time()
    pdata.cmr_data.extract_contours(visualize=visualize, align=align_segmentations, which_frames=which_frames)
    pdata.find_valves(valves, slices_3ch=valves_3ch_slice, visualize=visualize)
    pdata.generate_contours(which_frames=which_frames, visualize=visualize, sa_min_weight=0.5)
    contours_time = time() - contours_time

#%%
    fit_time = time()
    surfaces = pdata.fit_templates(which_frames=which_frames, parallelize=2)
    pdata.save_bv_surfaces(surfaces, mesh_subdivisions=0, which_frames=which_frames)
    fit_time = time() - fit_time

# #%%
#     from PatientData import FittedTemplate
#     from multiprocessing import Pool

#     def fit_template(frame):
#         print(f'Fitting template for frame {frame}...')
#         frame_prefix = f'{self.img2model_fldr}/frame{frame}_'

#         filename = frame_prefix + 'contours.txt'

#         template = FittedTemplate(frame_prefix, filemod='_mod2')
#         template.load_dataset(filename, weight_GP, load_control_points=load_control_points)

#         template.fit_template(weight_GP, low_smoothing_weight, transmural_weight, rv_thickness)
#         surface_mesh = template.bvmodel.get_template_mesh()

#         return frame, surface_mesh

#     self = pdata
#     weight_GP = 1
#     low_smoothing_weight = 10
#     transmural_weight = 20
#     rv_thickness = 3

#     self.surface_meshes = [None] * self.cmr_data.nframes
#     load_control_points = None

#     # Deal with which_frames
#     if which_frames is None:
#         which_frames = range(self.cmr_data.nframes)
#     elif isinstance(which_frames, int):
#         which_frames = [which_frames]
#     else:
#         which_frames = list(which_frames)

#     # Parallelize the fitting process using 2 cores
#     with Pool(processes=2) as pool:
#         results = pool.map(fit_template, which_frames)

#     # Store the results
#     for frame, surface_mesh in results:
#         self.surface_meshes[frame] = surface_mesh

#     print('Template fitting completed for all frames.')
