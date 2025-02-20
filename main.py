#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:26:30 2023

@author: Javiera Jilberto Vallejos
"""

import os
from PatientData import PatientData

if __name__ ==  '__main__':
    # Inputs
    patient = 'ZS-11'
    path = 'test_data/Images/'

    imgs = {'sa': path + 'SA',
            'la_2ch': path + 'LA_2CH',
            'la_3ch': path + 'LA_3CH',
            'la_4ch': path + 'LA_4CH'}


    segs = {'sa': path + 'sa_seg',
            'la_2ch': path + 'la_2ch_seg',
            'la_3ch': path + 'la_3ch_seg',
            'la_4ch': path + 'la_4ch_seg'}
    
    valves = {'la_2ch': path + 'LA_2CH_valves',
              'la_3ch': path + 'LA_3CH_valves',
              'la_4ch': path + 'LA_4CH_valves'}
    
    pdata = PatientData(imgs, path)
#     pdata.unet_generate_segmentations()
    import numpy as np
    which_frames = np.arange(13, 30).tolist()
    pdata.load_segmentations(segs)
    pdata.cmr_data.extract_contours(visualize=False, align=True, which_frames=which_frames)
    pdata.find_valves(valves, slices_3ch=[1], visualize=False)
    pdata.generate_contours(which_frames=which_frames, visualize=False)
    pdata.fit_templates(which_frames=which_frames, mesh_subdivisions=0, make_vol_mesh=True)
    lv_volume, rv_volume = pdata.calculate_chamber_volumes(which_frames=which_frames)
    print('LV volume: ', lv_volume)
    print('RV volume: ', rv_volume)
    lv_wall_volume, rv_wall_volume = pdata.calculate_wall_volumes(which_frames=which_frames)
    print('LV volume: ', lv_wall_volume)
    print('RV volume: ', rv_wall_volume)
