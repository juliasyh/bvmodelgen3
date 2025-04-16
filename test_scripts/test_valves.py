#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/11/27 11:03:44

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
from PatientData import PatientData
from niftiutils import readFromNIFTI
import valveutils as vu

# Inputs
path = '../test_data/Images/'

imgs = {'sa': path + 'SA',
        'la_2ch': path + 'LA_2CH',
        'la_3ch': path + 'LA_3CH',
        'la_4ch': path + 'LA_4CH'}


segs = {'sa': path + 'sa_seg',
        'la_2ch': path + 'la_2ch_seg',
        'la_3ch': path + 'la_3ch_seg',
        'la_4ch': path + 'la_4ch_seg'}

pdata = PatientData(imgs, path)
#     pdata.unet_generate_segmentations()
pdata.load_segmentations(segs)

# Deal with valves
segs = pdata.cmr_data.segs
imgs = pdata.cmr_data.imgs

# # 2CH
view = 'la_2ch'
seg = segs[view]
img = imgs[view]

mv_seg_points = vu.load_valve_nii(f'{path}/LA_2CH_valves', view)

slice = 0
mv_seg_points_slice = mv_seg_points[mv_seg_points[:,2]==slice][:,0:2]

mv_points, mv_centroids = vu.get_2ch_valve_points(seg, mv_seg_points_slice, slice=slice) 
vu.plot_valve_movement(img, seg, slice=0, valve_points={'mv': mv_points}, valve_centroids={'mv': mv_centroids})

# For the 3CH we only use one slice
view = 'la_3ch'
seg = segs[view]
img = imgs[view]

mv_seg_points, av_seg_points = vu.load_valve_nii(f'{path}/LA_3CH_valves', view)

slice = 1
mv_seg_points_slice = mv_seg_points[mv_seg_points[:,2]==slice][:,0:2]
av_seg_points_slice = av_seg_points[av_seg_points[:,2]==slice][:,0:2]

mv_points, mv_centroids, av_points, av_centroids = vu.get_3ch_valve_points(seg, slice=slice, 
                                                                           mv_seg_points=mv_seg_points_slice, 
                                                                           av_seg_points=av_seg_points_slice)

vu.plot_valve_movement(img, seg, slice=1, valve_points={'av': av_points, 'mv': mv_points},
                       valve_centroids={'av': av_centroids, 'mv': mv_centroids})


# 4CH
view = 'la_4ch'
seg = segs[view]
img = imgs[view]

mv_seg_points, tv_seg_points = vu.load_valve_nii(f'{path}/LA_4CH_valves', view)

slice = 0
mv_seg_points_slice = mv_seg_points[mv_seg_points[:,2]==slice][:,0:2]
tv_seg_points_slice = tv_seg_points[tv_seg_points[:,2]==slice][:,0:2]

mv_points, mv_centroids, tv_points, tv_centroids = vu.get_4ch_valve_points(seg, 
                                                                           mv_seg_points_slice, 
                                                                           tv_seg_points_slice,
                                                                           slice)

vu.plot_valve_movement(img, seg, slice=0, valve_points={'mv': mv_points, 'tv': tv_points}, 
                        valve_centroids={'mv': mv_centroids, 'tv': tv_centroids})
