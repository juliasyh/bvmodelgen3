#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:41:05 2023

@author: Javiera Jilberto Vallejos
"""

import os

import numpy as np
import nibabel as nib

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_cg

import plotutils as pu


def assign_translations_to_slice(nslices, slices):
    # Initializing dictionary to save translations
    translations = {}
    for view in nslices.keys():
        translations[view] = np.zeros([nslices[view],2])

    # Saving translations to array
    for n in range(len(slices)):
        trans = slices[n].accumulated_translation
        view = slices[n].view
        sn = slices[n].slice_number
        translations[view][sn] = trans

    return translations


def save_translations(prefix, nslices, slices):
    translations = assign_translations_to_slice(nslices, slices)

    # Saving translation to file
    for view in nslices.keys():
        fname = f'{prefix}{view}_translations'
        np.save(fname, translations[view])

    return translations


def slices2points(slices, which='lv'):
    points = []
    for n in range(len(slices)):
        points.append(slices[n].get_xyz_affine(which=which))
    return np.vstack(points)


def point_plane_intersection(points, p0, n):
    return (points-p0)@n


def intersect_two_slices(slice1, slice2, which):
    slice2_xyz = slice2.get_xyz_trans(which)

    # Finding the points that intersect with other slices
    dist_plane = point_plane_intersection(slice2_xyz, slice1.origin, slice1.normal)
    slice_ind = np.where(np.abs(dist_plane) < 0.5*slice2.pixdim)[0]

    return slice2_xyz[slice_ind]


def intersect_slice_normal(slc, translation, normal, origin, which):
    slc_xyz = slc.get_xyz_trans(which, translation)

    # Finding the points that intersect with other slices
    dist_plane = point_plane_intersection(slc_xyz, origin, normal)
    slice_ind = np.where(np.abs(dist_plane) < 0.5*slc.pixdim)[0]

    # For this function, I need it to return exactly two points
    points = slc_xyz[slice_ind]
    if len(points) > 2:
        dist = cdist(points, points)
        idx = np.argmax(dist)
        slice_ind = np.unravel_index(idx, dist.shape)
        return points[np.array(slice_ind)]
    else:
        return slc_xyz[slice_ind]

def intersect_two_slices_affine(slice1, slice2, which):
    slice2_xyz = slice2.get_xyz_affine(which)

    # Finding the points that intersect with other slices
    dist_plane = point_plane_intersection(slice2_xyz, slice1.origin, slice1.normal)
    slice_ind = np.where(np.abs(dist_plane) < 0.5*slice2.pixdim)[0]

    return slice2_xyz[slice_ind]

def intersect_two_boundaries(slice1, slice2, which):
    slice2_xyz = slice2.get_xyz_trans(which)

    # Finding the points that intersect with other slices
    dist_plane = point_plane_intersection(slice2_xyz, slice1.origin, slice1.normal)
    slice_ind = np.where(np.abs(dist_plane) < 0.5*slice2.pixdim)[0]

    return slice2_xyz[slice_ind]



def slice_intersection_error(translation, ind, slices):
    sij = []
    contours = ['lvendo', 'lvepisep', 'rvendo']

    slc = slices[ind]
    slc_xyz = {}
    for contour in contours:
        if (contour == 'rvendo') and (not slc.has_rv): continue
        slc_xyz[contour] = slc.get_xyz_trans(contour, translation)  # precompute this

    for m in range(len(slices)):
        weight=1
        # if m == ind: continue   # skip the same slice
        if slc.view == 'sa' and slices[m].view == 'sa': continue # 'sa slices do not intersect'
        if slc.view == 'la' and slices[m].view == 'la': weight=5

        # For each contour type I compute the error
        for contour in slc_xyz.keys():
            if (contour == 'rvendo') and (not slices[m].has_rv): continue
            intersect_points = intersect_two_slices(slc, slices[m], contour)

            tree = KDTree(slc_xyz[contour])
            distance, _ = tree.query(intersect_points)

            sij.append(distance*weight)

    values = np.concatenate(sij)

    error = np.sum((values)**2)/len(values)

    return error


def optimize_stack_translation(slices, nit=100):
    for i in range(nit):
        translations = []
        for n in range(len(slices)):
            sol = fmin_cg(slice_intersection_error, np.zeros(2), args=(n,slices), disp=False)
            translations.append(sol)
            slices[n].accumulated_translation += sol

        translations = np.vstack(translations)
        trans_norm = np.linalg.norm(translations)
        print('Iteration {:d}, transform norm: {:2.3e}'.format(i, trans_norm))
        if trans_norm < 1e-2:
            break


def identify_base_apex(contour):
    length = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    ind = np.argmax(length)

    point1 = contour[ind]
    point2 = contour[ind+1] if ind < len(contour)-1 else contour[0]
    base = (point1 + point2) / 2

    apex = contour[np.argmax(np.linalg.norm(contour - base, axis=1))]
    return base, apex


def slice_intersection_error2(translation, ind, slices):
    sij = []
    contours = ['lvendo', 'lvepisep', 'rvendo']

    slc = slices[ind]
    slc_xyz = {}
    for contour in contours:
        if (contour == 'rvendo') and (not slc.has_rv): continue
        slc_xyz[contour] = slc.get_xyz_trans(contour, translation)  # precompute this

    if ('la' in slc.view):
        # Find the base and apex of the contour
        base, apex = identify_base_apex(slc_xyz['lvendo'])
        la_length = np.linalg.norm(base-apex)

    la_slices = []
    sa_slices = []
    for m in range(len(slices)):
        # Save SA/LA slices to loop after
        if 'la' in slices[m].view:
            la_slices.append(slices[m])
        if 'sa' in slices[m].view:
            sa_slices.append(slices[m])

        weight=1
        # if m == ind: continue   # skip the same slice
        if slc.view == 'sa' and slices[m].view == 'sa': continue # 'sa slices do not intersect'

        # For each contour type I compute the error
        for contour in slc_xyz.keys():
            if (contour == 'rvendo') and (not slices[m].has_rv): continue
            intersect_points = intersect_two_slices(slc, slices[m], contour)

            tree = KDTree(slc_xyz[contour])
            distance, _ = tree.query(intersect_points)

            if ('la' in slc.view) and ('la' in slices[m].view): 
                apex_dist = intersect_points - apex
                apex_dist = np.linalg.norm(apex_dist, axis=1)
                apex_dist_norm = apex_dist/la_length
                
                apex_dist_norm = np.clip(apex_dist_norm, 0, 1)
                weight = (1 - apex_dist_norm)

            sij.append(distance*weight)

    # If the slice is SA, then we compute the distance to the neighboring slices
    # at a perpendicular long axis plane
    if 'sa' in slc.view:
        sa_normal = slc.normal
        sa_origin = np.mean(slc_xyz['lvendo'], axis=0)
        for la_slice in la_slices:
            cross_normal = np.cross(la_slice.normal, sa_normal)
            cross_normal = cross_normal/np.linalg.norm(cross_normal)

            for contour in slc_xyz.keys():
                if (contour == 'rvendo'): continue # RV is weird
                slice_points = intersect_slice_normal(slc, translation, cross_normal, sa_origin, which=contour)
                if ind == 0:
                    toiterate = [ind+1]
                elif ind == (len(sa_slices)-1):
                    toiterate = [ind-1]
                else:
                    toiterate = [ind-1, ind+1]

                points = []
                # Compute intersection points in each slice
                for i, n in enumerate(toiterate):
                    pts = intersect_slice_normal(slices[n], np.zeros(2), cross_normal, sa_origin, which=contour)
                    if len(pts) > 1:
                        points.append(pts)
                if len(points) == 0:
                    continue


                # la_slice_xyz = la_slice.get_xyz_trans(contour, translation)
                # sa_slice_xyz = slc.get_xyz_trans(contour, translation)
                # fig = plt.figure(23, clear=True)
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(slc_xyz[contour][:,0],slc_xyz[contour][:,1],slc_xyz[contour][:,2])
                # ax.scatter(la_slice_xyz[:,0],la_slice_xyz[:,1],la_slice_xyz[:,2])
                # ax.scatter(slc_xyz[:,0],slc_xyz[:,1],slc_xyz[:,2])
                # ax.scatter(origin[0],origin[1],origin[2])
                # ax.scatter(origin[0]+normal[0],origin[1]+normal[1],origin[2]+normal[2])
                # ax.scatter(points[:,0],points[:,1],points[:,2], s=60)

                # Loop over the points and compute distance to the slice on top
                for i in range(len(points)):
                    distance = cdist(slice_points, points[i])
                    distance = np.min(distance, axis=1)
                    sij.append(distance*weight)

    values = np.concatenate(sij)

    error = np.sum((values)**2)/len(values)

    return error


def slice_intersection_error3(translation, ind, slices):
    sij = []
    contours = ['lvendo', 'lvepisep', 'rvendo']

    slc = slices[ind]
    slc_xyz = {}
    for contour in contours:
        if (contour == 'rvendo') and (not slc.has_rv): continue
        slc_xyz[contour] = slc.get_xyz_trans(contour, translation)  # precompute this

    la_slices = []
    sa_slices = []
    for m in range(len(slices)):
        # Save SA/LA slices to loop after
        if 'la' in slices[m].view:
            la_slices.append(slices[m])
        if 'sa' in slices[m].view:
            sa_slices.append(slices[m])
        weight=1
        # if m == ind: continue   # skip the same slice
        if slc.view == 'sa' and slices[m].view == 'sa': continue # 'sa slices do not intersect'
        # For each contour type I compute the error
        for contour in slc_xyz.keys():
            if (contour == 'rvendo') and (not slices[m].has_rv): continue
            intersect_points = intersect_two_slices(slc, slices[m], contour)

            tree = KDTree(slc_xyz[contour])
            distance, _ = tree.query(intersect_points)

            sij.append(distance*weight)

    # If the slice is SA, then we compute the distance to the neighboring slices
    # at a perpendicular long axis plane
    if 'sa' in slc.view:
        sa_normal = slc.normal
        sa_origin = np.mean(slc_xyz['lvendo'], axis=0)
        for la_slice in la_slices:
            cross_normal = np.cross(la_slice.normal, sa_normal)
            cross_normal = cross_normal/np.linalg.norm(cross_normal)

            for contour in slc_xyz.keys():
                if (contour == 'rvendo'): continue # RV is weird
                slice_points = intersect_slice_normal(slc, translation, cross_normal, sa_origin, which=contour)
                if ind == 0:
                    toiterate = [ind+1]
                elif ind == (len(sa_slices)-1):
                    toiterate = [ind-1]
                else:
                    toiterate = [ind-1, ind+1]

                points = []
                # Compute intersection points in each slice
                for i, n in enumerate(toiterate):
                    pts = intersect_slice_normal(slices[n], np.zeros(2), cross_normal, sa_origin, which=contour)
                    if len(pts) > 1:
                        points.append(pts)
                if len(points) == 0:
                    continue

                # Loop over the points and compute distance to the slice on top
                for i in range(len(points)):
                    distance = cdist(slice_points, points[i])
                    distance = np.min(distance, axis=1)
                    sij.append(distance*weight)

    # If the slice is LA, then we compute the distance to the neighboring slices
    # at a perpendicular short axis plane
    if 'la' in slc.view:
        la_normal = slc.normal
        la_origin = np.mean(slc_xyz['lvendo'], axis=0)
        for sa_slice in sa_slices:
            cross_normal = np.cross(sa_slice.normal, la_normal)
            cross_normal = cross_normal/np.linalg.norm(cross_normal)

            for contour in slc_xyz.keys():
                if (contour == 'rvendo'): continue # RV is weird
                slice_points = intersect_slice_normal(slc, translation, cross_normal, la_origin, which=contour)
                if ind == len(sa_slices):
                    toiterate = [ind+1]
                elif ind == (len(slices)-1):
                    toiterate = [ind-1]
                else:
                    toiterate = [ind-1, ind+1]

                points = []
                # Compute intersection points in each slice
                for i, n in enumerate(toiterate):
                    pts = intersect_slice_normal(slices[n], np.zeros(2), cross_normal, la_origin, which=contour)
                    if len(pts) > 1:
                        points.append(pts)
                if len(points) == 0:
                    continue
                points = np.stack(points, axis=0)
                # Loop over the points and compute distance to the slice on top
                for i in range(len(points)):
                    distance = cdist(slice_points, points[i])
                    distance = np.min(distance, axis=1)
                    sij.append(distance*weight)

    values = np.concatenate(sij)

    error = np.sum((values)**2)/len(values)

    return error


def optimize_stack_translation2(slices, which='both', nit=100):
    for i in range(nit):
        translations = []
        for n in range(len(slices)):
            sol = fmin_cg(slice_intersection_error2, np.zeros(2), args=(n,slices), disp=False)
            translations.append(sol)
            slices[n].accumulated_translation += sol

        translations = np.vstack(translations)
        trans_norm = np.linalg.norm(translations)
        print('Iteration {:d}, transform norm: {:2.3e}'.format(i, trans_norm))
        if trans_norm < 1e-2:
            break

def optimize_stack_translation3(slices, nit=100):
    for i in range(nit):
        translations = []
        for n in range(len(slices)):
            sol = fmin_cg(slice_intersection_error3, np.zeros(2), args=(n,slices), disp=False)
            translations.append(sol)
            slices[n].accumulated_translation += sol

        translations = np.vstack(translations)
        trans_norm = np.linalg.norm(translations)
        print('Iteration {:d}, transform norm: {:2.3e}'.format(i, trans_norm))
        if trans_norm < 1e-2:
            break


def get_slice_intersection_points(ind, slices):
    points = []
    slc = slices[ind]
    for m in range(len(slices)):
        if m == ind: continue   # skip the same slice
        if slc.view == 'sa' and slices[m].view == 'sa': continue  # Sa slices do not intersect
        intersect_points = intersect_two_slices(slc, slices[m], 'lv')
        points.append(intersect_points)

    return np.vstack(points)


# https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
def fit_circle_to_points(xy):
    from scipy.optimize import leastsq

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    x, y = xy.T
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = leastsq(f, center_estimate, Dfun=Df, col_deriv=True)

    Ri = calc_R(*center)
    R = Ri.mean()

    return center, R

def find_SA_initial_guess(slices):  # TODO use contours
    for n in range(len(slices)):
        slc = slices[n]
        if slc.view != 'sa': continue
        points = get_slice_intersection_points(n, slices)

        A, t = nib.affines.to_matvec(slc.affine)
        points_ij = np.linalg.solve(A, (points-t).T).T[:,0:2]

        # If no intersection was found, continue
        if len(points_ij) == 0:
            continue

        # Centroids in xyz
        centroid_ij, r = fit_circle_to_points(points_ij)
        slc_centroid = np.mean(slc.get_xyz_trans('lv'), axis=0)

        # Centroid in ijk
        slc_centroid_ij = np.linalg.solve(A, slc_centroid-t)[0:2]

        slc.accumulated_translation += centroid_ij - slc_centroid_ij


def slice_intersection_error_affine(affine, ind, slices):
    sij = []
    contours = ['lvendo', 'lvepi', 'rvendo']

    slc = slices[ind]
    slc_xyz = {}
    for contour in contours:
        if (contour == 'rvendo') and (not slc.has_rv): continue
        slc_xyz[contour] = slc.get_xyz_affine(contour, affine)  # precompute this
    for m in range(len(slices)):
        weight=1
        # if (m == ind) and (slc.view == 'sa'): weight=100 # continue   # skip the same slice
        if slc.view == 'sa' and slices[m].view == 'sa': continue # 'sa slices do not intersect'
        if slc.view == 'la' and slices[m].view == 'la': weight=1
        for contour in slc_xyz.keys():
            if (contour == 'rvendo') and (not slices[m].has_rv): continue
            intersect_points = intersect_two_slices_affine(slc, slices[m], contour)

            tree = KDTree(slc_xyz[contour])
            distance, _ = tree.query(intersect_points)

            sij.append(weight*distance)

    values = np.concatenate(sij)

    error = np.sum((values)**2)/len(values)

    return error

def optimize_stack_affine(slices, nit=10):
    for i in range(nit):
        affines = []
        for n in reversed(range(len(slices))):
            sol = fmin_cg(slice_intersection_error_affine, np.zeros(6), args=(n,slices), disp=False)
            affines.append(sol)
            slices[n].accumulated_translation += sol[4:]
            slices[n].accumulated_matrix = (np.eye(2) + sol[0:4].reshape([2,2]))@slices[n].accumulated_matrix

        affines = np.vstack(affines)
        trans_norm = np.linalg.norm(affines[4:])
        print('Iteration {:d}, transform norm: {:2.3e}'.format(i, trans_norm))
        if trans_norm < 1e-1:
            break



def plot_slice_intersection(ind, slices, use_cum_trans=True, use_cum_affine=True):
    points = []

    slc = slices[ind]
    slc_xyz = slc.get_xyz_affine('lv', use_cum_trans=use_cum_trans, use_cum_affine=use_cum_affine)
    for m in range(len(slices)):
        # if m == ind: continue   # skip the same slice
        if slc.view == 'sa' and slices[m].view == 'sa': continue # 'sa slices do not intersect'
        slice2 = slices[m]
        slice2_xyz = slice2.get_xyz_affine('lv', use_cum_trans=use_cum_trans, use_cum_affine=use_cum_affine)

        # Finding the points that intersect with other slices
        dist_plane = point_plane_intersection(slice2_xyz, slc.origin, slc.normal)
        slice_ind = np.where(np.abs(dist_plane) < 0.5*slice2.pixdim)[0]

        intersect_points = slice2_xyz[slice_ind]
        points.append(intersect_points)

    points = np.vstack(points)

    fig = pu.show_point_cloud(slc_xyz, opacity=0.5, size=5, color='blue')
    fig = pu.show_point_cloud(points, opacity=0.5, size=5, fig=fig, color='red')
    fig.show()


