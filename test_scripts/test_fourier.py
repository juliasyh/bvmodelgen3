#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/10 16:23:24

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import matplotlib.pyplot as plt

import meshio as io
from tqdm import tqdm

print('Loading data...')
xyzs = []
for i in tqdm(range(30)):
    surface = io.read(f'test_data/Images/img2model/frame{i}_bv_surface.stl')
    xyzs.append(surface.points)

xyzs = [xyzs[-2], xyzs[-1]] + xyzs + [xyzs[0], xyzs[1]]
xyzs = np.stack(xyzs)

#%% Calculate displacements
disp = xyzs - xyzs[2]
flatten_disp = disp.reshape(disp.shape[0], -1)
flatten_disp = np.swapaxes(flatten_disp, 0, 1)

fourier_transform = np.fft.fft(flatten_disp, axis=1)

# Smooth the displacement using the Fourier space
cutoff = 0.2  # Define a cutoff frequency
freqs = np.fft.fftfreq(fourier_transform.shape[1])
fourier_transform[:,np.abs(freqs) > cutoff] = 0

# Inverse Fourier transform to get the smoothed displacement
smoothed_disp = np.fft.ifft(fourier_transform, axis=1).real
smoothed_disp_reshape = smoothed_disp.reshape(flatten_disp.shape)

#%% Save displacements to mesh
smoothed_disp_reshape = np.swapaxes(smoothed_disp_reshape, 0, 1)
smoothed_disp_reshape = smoothed_disp_reshape.reshape(disp.shape)
for i in range(2,32):
    surface.points = points=xyzs[2] + smoothed_disp_reshape[i]
    io.write(f'test_data/Images/img2model/frame{i-2}_bv_surface_smoothed.stl', surface)

# #%%
# # Plot the displacements

# print('Plotting...')
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # Plot the displacements
# color=['r', 'g', 'b']
# ax1.plot(flatten_disp[0], '--', alpha=0.5, color=color[0])
# ax1.plot(smoothed_disp[0], color=color[0])
# ax1.set_xlabel('Frame')
# ax1.set_ylabel('Displacement')
# ax1.set_title('Displacement of Point 1 Over Time')
# ax1.grid(True)

# # Plot the Fourier transform
# ax2.plot(freqs, np.abs(fourier_transform[0]))
# ax2.set_xlabel('Frequency')
# ax2.set_ylabel('Amplitude')
# ax2.set_title('Fourier Transform of Displacement')

# plt.tight_layout()
# plt.show()

# #%% Use fenicsx to smooth in space
# import dolfinxio as io
# vol_ien = np.load('src/bvfitting/template/volume_template_ien.npy')
# points = xyzs[2]
