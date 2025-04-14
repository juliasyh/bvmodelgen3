#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/10 17:46:43

@author: Javiera Jilberto Vallejos 
'''

from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
from niftiutils import readFromNIFTI, saveInterpNIFTI
from tqdm import tqdm
from matplotlib.widgets import Slider
from imgutils import interpolate_scans

img, affine, pixdim, header = readFromNIFTI('../test_data/Images/SA', return_header=True)

nframes_og = img.shape[3]
nframes = 45

timepoints_og = np.linspace(0, 1, nframes_og)
timepoints = np.linspace(0, 1, nframes)

interp_img = interpolate_scans(img, nframes)

# Fix slice duration
tcycle = header['slice_duration']*nframes_og

# Save the interpolated image
saveInterpNIFTI('../test_data/Images/SA', '../test_data/Images/SA_interp.nii.gz', interp_img, tcycle/nframes)


#%%
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Initial time index
time_idx = 0

# Display the original image
slice_idx = 5
original_img_plot = axes[0].imshow(img[:, :, slice_idx, time_idx], cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the interpolated image
interpolated_img_plot = axes[1].imshow(interp_img[:, :, slice_idx, time_idx], cmap='gray')
axes[1].set_title('Interpolated Image')
axes[1].axis('off')

# Add a slider for time
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time', 0, 1, valinit=0, valstep=1/(nframes-1))

# Update function for the slider
def update(val):
    time_idx = int(val * (nframes_og - 1))
    original_img_plot.set_data(img[:, :, slice_idx, time_idx])
    time_idx = int(val * (nframes - 1))
    interpolated_img_plot.set_data(interp_img[:, :, slice_idx, time_idx])
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()