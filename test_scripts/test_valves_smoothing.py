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
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist 

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


# Make points coherent
for i in range(1, 30):
    dist = cdist(mv_points[i], mv_points[i-1])
    mv_points[i] = mv_points[i][np.argmin(dist, axis=1)]
    

# Get mean normal
mean_normal = np.mean(mv_normals, axis=0)
mean_normal = mean_normal/np.linalg.norm(mean_normal)

# Smooth centroid displacement
mv_cent_disp = mv_centroids - mv_centroids[0]
mv_cent_disp_ext = np.vstack((mv_cent_disp, mv_cent_disp, mv_cent_disp)) 

mv_cent_disp_smooth = get_smooth_displacements(mv_cent_disp_ext, cutoff=0.2)
mv_cent_pos = mv_cent_disp_smooth + mv_centroids[0]

plt.figure()
plt.plot(mv_cent_disp_ext)
plt.plot(mv_cent_disp_smooth)

# Grab position respect the centroid
mv_points_cent = mv_points - mv_centroids[:, np.newaxis, :]
# Calculate the angle using the first point as 0 degrees.
mv_points_zpos = np.dot(mv_points_cent, mean_normal)
mv_points_plane_vector = mv_points_cent - mv_points_zpos[:, :, np.newaxis] * mean_normal
mv_points_rad = np.linalg.norm(mv_points_plane_vector, axis=2)
mv_points_plane_vector_norm = mv_points_plane_vector / np.linalg.norm(mv_points_plane_vector, axis=2)[:, :, None]
aux_vector = mv_points_plane_vector[0, 0] / np.linalg.norm(mv_points_plane_vector[0, 0])
cross_aux_vector = np.cross(mean_normal, aux_vector)

# Calculate the angle using arctan2
mv_points_angle = np.arctan2(
    np.sum(mv_points_plane_vector * cross_aux_vector, axis=2),
    np.sum(mv_points_plane_vector * aux_vector, axis=2),
)

# Smooth z and rad displacements
mv_points_zdisp = mv_points_zpos - mv_points_zpos[0]
mv_points_rad_disp = mv_points_rad - mv_points_rad[0]
mv_points_zdisp_ext = np.vstack((mv_points_zdisp, mv_points_zdisp, mv_points_zdisp))
mv_points_rad_disp_ext = np.vstack((mv_points_rad_disp, mv_points_rad_disp, mv_points_rad_disp))
mv_points_zdisp_smooth = get_smooth_displacements(mv_points_zdisp_ext, cutoff=0.15)
mv_points_rad_disp_smooth = get_smooth_displacements(mv_points_rad_disp_ext, cutoff=0.15)
mv_points_angle_smooth = get_smooth_displacements(mv_points_angle, cutoff=0.15)

# Reconstruct points
mv_points_smooth = np.zeros_like(mv_points)
for i in range(30):
    angle_smooth = mv_points_angle_smooth[i]
    k = mean_normal
    v = aux_vector
    cos_angle = np.cos(angle_smooth)
    sin_angle = np.sin(angle_smooth)
    plane_vector = (v * cos_angle[:, np.newaxis] +
                         np.cross(k, v) * sin_angle[:, np.newaxis] +
                         k * np.dot(k, v) * (1 - cos_angle[:, np.newaxis]))
    
    mv_points_rad_pos = mv_points_rad_disp_smooth[i] + mv_points_rad[0]
    mv_points_z_pos = mv_points_zdisp_smooth[i] + mv_points_zpos[0]
    mv_points_smooth[i] = (plane_vector * mv_points_rad_pos[:, np.newaxis] 
                        + mean_normal * mv_points_z_pos[:, np.newaxis]
                        + mv_cent_pos[i])





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot
sc = ax.scatter(mv_points[0][:, 0], mv_points[0][:, 1], mv_points[0][:, 2], c='b')
sc = ax.scatter(mv_points_smooth[0][:, 0], mv_points_smooth[0][:, 1], mv_points_smooth[0][:, 2], c='r')
# ax.set_xlim((-40, 0))
# ax.set_ylim((30, 60))
# ax.set_zlim((-20, 30))

ax.scatter(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], c='k', alpha=0.1)

# Slider
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, 29, valinit=0, valstep=1)

def update(val):
    frame = int(slider.val)
    ax.clear()
    ax.scatter(mv_points[frame][:, 0], mv_points[frame][:, 1], mv_points[frame][:, 2], c='b')
    ax.scatter(mv_points_smooth[frame][:, 0], mv_points_smooth[frame][:, 1], mv_points_smooth[frame][:, 2], c='r')
    ax.scatter(mesh.points[:,0], mesh.points[:,1], mesh.points[:,2], c='k', alpha=0.1)
    plt.draw()

slider.on_changed(update)


plt.show()