#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/24 18:01:26

@author: Javiera Jilberto Vallejos 
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

img = nib.load('test_data/Images/LA_4CH.nii')        # 4D stack
img_data = img.get_fdata()
zooms = img.header.get_zooms()
dt = zooms[-1]  # temporal resolution in miliseconds


seg = nib.load('test_data/Images/la_4ch_seg.nii.gz') # 4D stack
seg_data = seg.get_fdata()


nframes = img_data.shape[-1]
assert nframes == seg_data.shape[-1], 'Number of frames in image and segmentation do not match'

framerate = 1 / (dt/1000)
print(f'Framerate for real-time video: {framerate} frames per second')

output_dir = 'test_data/pngs/'
os.makedirs(output_dir, exist_ok=True)

for frame in range(nframes):
    fig, ax = plt.subplots()
    ax.imshow(img_data[:, :, 0, frame], cmap='gray')
    seg_overlay = np.ma.masked_where(seg_data[:, :, 0, frame] == 0, seg_data[:, :, 0, frame])
    ax.imshow(seg_overlay, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(f'{output_dir}/frame_{frame:03d}.png', bbox_inches='tight', pad_inches=0, dpi=180)
    plt.close(fig)