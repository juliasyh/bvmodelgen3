#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/12 11:48:49

@author: Javiera Jilberto Vallejos 
'''

import numpy as np
import meshio as io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def get_normal_plane_svd(points):   # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid

def get_smooth_displacements(points, cutoff=0.4):
    points_fft = np.fft.fft(points, axis=0)
    frequencies = np.fft.fftfreq(points.shape[0])


    # Filter out high frequencies
    points_fft[np.abs(frequencies) > cutoff] = 0

    plt.plot(frequencies, np.abs(points_fft[:, 0]))

    # Inverse FFT to get the smoothed signal
    points_smoothed = np.fft.ifft(points_fft, axis=0).real
    return points_smoothed

path = 'test_data/Images/img2model/'

mv_points = np.zeros((30, 6, 3))
mv_centroids = np.zeros((30, 3))
mv_normals = np.zeros((30, 3))
for i in range(30):
    mesh = io.read(f'{path}/frame{i}_contours.vtu')
    mv_points[i] = mesh.points[mesh.point_data['label'] == 5]
    mv_normals[i], mv_centroids[i] = get_normal_plane_svd(mv_points[i])


# Make sure all normals are pointing in the same direction
for i in range(1, 30):
    if np.dot(mv_normals[i], mv_normals[i-1]) < 0:
        mv_normals[i] = -mv_normals[i]

# Get mean normal
mean_normal = np.mean(mv_normals, axis=0)

# Smooth centroid displacement
mv_cent_disp = mv_centroids - mv_centroids[0]
mv_cent_disp_ext = np.vstack((mv_cent_disp, mv_cent_disp, mv_cent_disp)) 

mv_cent_disp_smooth = get_smooth_displacements(mv_cent_disp_ext, cutoff=0.2)

plt.figure()
plt.plot(mv_cent_disp_smooth)

# Calculate out-of-plane displacement and radial displacement




# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Initial plot
# sc = ax.scatter(mv_points[0][:, 0], mv_points[0][:, 1], mv_points[0][:, 2], c=np.arange(6))
# ax.set_xlim((-40, 0))
# ax.set_ylim((30, 60))
# ax.set_zlim((-20, 30))

# # Slider
# ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Frame', 0, 29, valinit=0, valstep=1)

# def update(val):
#     frame = int(slider.val)
#     ax.clear()
#     ax.scatter(mv_points[frame][:, 0], mv_points[frame][:, 1], mv_points[frame][:, 2], c=np.arange(6))
#     plt.draw()

# slider.on_changed(update)


# plt.show()