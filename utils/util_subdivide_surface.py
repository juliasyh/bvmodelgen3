#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/05 12:56:19

@author: Javiera Jilberto Vallejos 
'''


from PatientData import PatientData
from time import time

if __name__ ==  '__main__':

    ##############################################################
    ######  USER INPUTS AND OPTIONS ##############################
    ##############################################################

    # Options
    interpolate_segmentations = 'max'  # ('max', 'min', int, None) means to interpolate the segmentations to the max or min number of frames, 
                                       # or to a specific number of frames. None means not to interpolate.
    
    # Inputs
    output_path = 'DSP-3.5001/Images/'   # where all the output will be saved
    imgs_path = 'DSP-3.5001/Images/'    # Dummy variable to define common paths, not truly needed if you define the paths directly

    imgs = {'sa': imgs_path + 'SA',
            'la_2ch': imgs_path + 'LA_2CH',
            'la_3ch': imgs_path + 'LA_3CH',
            'la_4ch': imgs_path + 'LA_4CH'}

    # Which frames to process
    which_frames = [0]  # None means all frames. Remember Python starts with 0!

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
    

    # Initialize the PatientData object
    pdata = PatientData(imgs, output_path)
    pdata.load_segmentations(segs)

    # Generate surfaces
    surfaces = pdata.load_surfaces(which_frames=which_frames)
    pdata.save_bv_surfaces(surfaces, mesh_subdivisions=2, prefix='subdiv_', which_frames=which_frames)