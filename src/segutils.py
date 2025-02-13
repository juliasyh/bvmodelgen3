#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/11 16:46:49

@author: Javiera Jilberto Vallejos 
'''

import numpy as np

from skimage import measure, morphology
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from shapely.geometry import Polygon
import csv
        

def correct_labels(seg, labels):
    new_seg = np.copy(seg)
    for i, which in enumerate(['lvbp', 'lv', 'rv']):
        vals = labels[which]
        if type(vals) == list:
            for v in vals:
                new_seg[seg == v] = i+1
        else:
            new_seg[seg == vals] = i+1

    return new_seg


def remove_holes_islands(mask, irregMaxSize=20):
    # Remove isolated small objects
    cleanmask = morphology.remove_small_objects(np.squeeze(mask), min_size=irregMaxSize, connectivity=2)

    # Close small holes (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(cleanmask)
    seed[1:-1, 1:-1] = cleanmask.max()
    cleanmask = np.squeeze(morphology.reconstruction(seed, cleanmask, method='erosion'))

    return cleanmask


def get_number_of_objects(mask):
    regions = measure.label(mask)
    return len(np.unique(regions)) - 1

def get_mask_eccentricity(mask):
    props = measure.regionprops(mask.astype(int))[0]
    return props.eccentricity

def getContoursFromMask(maskSlice, irregMaxSize):
    '''
    maskSlice is a 2D ndarray, i.e it is a m x n ndarray for some m, n. This function returns a m x 2 ndarray, where
    each row in the array represents a point in the contour around maskSlice.
    '''

    # First, clean the mask.

    # Remove irregularites with fewer than irregMaxSize pixels.Note, the "min_size" parameter to this function is
    # incorrectly named, and should really be called "max_size".
    maskSlice = morphology.remove_small_objects(np.squeeze(maskSlice), min_size=irregMaxSize, connectivity=2)  # Might have to come back and see if connectivity = 2 was the right choice

    # Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(maskSlice)
    seed[1:-1, 1:-1] = maskSlice.max()
    maskSlice = np.squeeze(morphology.reconstruction(seed, maskSlice, method='erosion'))

    # get contours
    lst = measure.find_contours(maskSlice, level = .5)

    # This "else" in the below is the reason this function is necessary; np.stack() does not accept an empty list.
    return np.vstack(lst) if not len(lst) == 0 else np.array([])



def pointDistances(points):
    '''
    "points" is an m x 2 ndarray for some m.

    Returns an m x 1 ndarray in which the ith entry is the distance from the ith point in "points" to the (i + 1)st point.
    The last entry is the distance from the last point to the first point.
    '''

    numPts = points.shape[0]
    distances = []
    for i in range(0, numPts): # i covers the range {0, ..., numPts - 1} since numPts isn't included
        if i == numPts - 1:  # If on last point, compare last point with first point
            distances.append(np.linalg.norm(points[i, :] - points[0, :]))
        else:
            distances.append(np.linalg.norm(points[i + 1, :] - points[i, :]))

    return np.array(distances)

def deleteHelper(arr, indices, axis = 0):
    '''
    Helper function for deleting parts of ndarrays that, unlike np.delete(), works when either "arr" or "indices" is empty.
    (np.delete() does not handle the case in which arr is an empty list or an empty ndarray).

    "arr" may either be a list or an ndarray.
    '''

    def emptyNdarrayCheck(x):
        return type(x) is np.ndarray and ((x == np.array(None)).any() or x.size == 0)

    def emptyListCheck(x):
        return type(x) is list and len(x) == 0

    if emptyNdarrayCheck(arr) or emptyListCheck(arr):
        return arr

    # np.delete() does not work as expected if indices is an empty ndarray. This case fixes that.
    # (If indices is an empty list, np.delete() works as expected, so there is no need to use emptyListCheck() here).
    if emptyNdarrayCheck(indices):
        return arr

    return np.delete(arr, indices, axis)

def sharedRows(arr1, arr2):
    '''
    "arr1" and "arr2" must be m x n ndarrays for some m and n. (So they can't be m x n x s ndarrays).

    Returns the list [_sharedRows, sharedIndicesArr1, sharedIndicesArr2].
    "_sharedRows" is a matrix whose rows are those that are shared by "arr1" and "arr2".
    "sharedIndicesArr1" is a list of the row-indices in arr1 whose rows appear (not necessarily with the same indices) in
    arr2.
    "sharedIndicesArr2" is a similar list of row-indices that pertains to "arr2".
    '''

    if arr1 is None or arr1.size == 0 or arr2 is None or arr2.size == 0: #If either array is empty, return a list containing three empty ndarrays.
        return[np.array([]), np.array([]), np.array([])]

    if arr1.shape[1] != arr2.shape[1]:
        raise ValueError("Arrays must have same number of columns.")

    # Get the indices of the shared rows.
    sharedIndicesArr1 = []
    sharedIndicesArr2 = []

    for i in range(0, arr1.shape[0]):
        for j in range(0, arr2.shape[0]):
            if i in sharedIndicesArr1 or j in sharedIndicesArr2:
                continue
            elif np.all(arr1[i, :] == arr2[j, :]): #If (ith row in arr1) == (jth row in arr2)
                sharedIndicesArr1.append(i)
                sharedIndicesArr2.append(j)

    # Use these indices to build the matrix of shared rows.
    _sharedRows = arr1[sharedIndicesArr1, :]

    # Convert the lists of sharedIndices to ndarrays.
    sharedIndicesArr1 = np.array(sharedIndicesArr1)
    sharedIndicesArr2 = np.array(sharedIndicesArr2)

    return [_sharedRows, sharedIndicesArr1, sharedIndicesArr2]


def getRVinsertIndices(points):
    '''
    "points" is a m1 x 2 ndarray for some m1.
    Returns an 1 x m2 ndarray, for some m2, containing the indices of endoRVFWContours that correspond to the RV insert points.
    '''

    distances = pointDistances(points)
    upperThreshold = np.mean(distances) + 3 * np.std(distances, ddof = 1) # We need to use ddof = 1 to use Bessel's correction (so we need it to get the same std as is calculated in MATLAB).
    largeDistExists = np.any(distances > upperThreshold)

    # Find the index (in "distances") of the point that is furthest from its neighbor. Return an ndarray consisting of
    # this point and *its* neighbor.
    if largeDistExists != 0:
        largestDistIndex = np.argmax(distances)
        if largestDistIndex == len(points) - 1: # if the point furthest from its neighbor is the last point...
            return np.array([0, largestDistIndex]) #the neighbor to largestDistIndex is 0 in this case
        else:
            return np.array([largestDistIndex, largestDistIndex + 1])
    else:
        return np.array([])


def getLAinsert(inserts, lv_endo_points, lv_epi_points):
    # The lv_endo and lv_epi at this point still have the base.
    # First we find a point in the base by looking at the points with the shortest distance between lv_endo and lv_epi
    tree = KDTree(lv_endo_points)
    dist, _ = tree.query(lv_epi_points)
    base_point = lv_endo_points[np.argmin(dist)]

    # Now we find the point of the rv_insert that's farthest from the base
    dist = cdist(inserts, [base_point])
    
    return inserts[np.argmax(dist)]


def remove_base_nodes(contours, apex=None, mv_centroid=None, min_length=15):
    # Grab la contours
    la_contours = [ctr for ctr in contours if ('la' in ctr.view) and ('endo' in ctr.ctype or 'epi' in ctr.ctype) and ('lv' in ctr.ctype or 'rv' in ctr.ctype)]
    sa_contours = [ctr for ctr in contours if ('sa' in ctr.view) and ('endo' in ctr.ctype or 'epi' in ctr.ctype) and ('lv' in ctr.ctype or 'rv' in ctr.ctype)]
    if apex is None:
        try:
            apex = [ctr.points for ctr in contours if ctr.ctype == 'apexepi'][0]
        except:
            apex = [ctr.points for ctr in contours if ctr.ctype == 'apexendo'][0]
    else:
        apex = apex
    if mv_centroid is None:
        mv_points = np.vstack([ctr.points for ctr in contours if ctr.ctype == 'mv'])
        mv_centroid = np.mean(mv_points, axis=0)
    else:
        mv_centroid = mv_centroid

    sa_vector = sa_contours[0].normal

    la_length = np.dot(mv_centroid - apex, sa_vector)
    if la_length < 0:
        sa_vector = -sa_vector
        la_length = -la_length

    # Compute long axis distance for la contours
    for ctr in la_contours:
        points = ctr.points
        z_coord = np.dot(points - apex, sa_vector)/la_length

        len_base_nodes = 0
        long_cut = 0.85
        while len_base_nodes < min_length:
            base_nodes = np.where(z_coord > long_cut)[0]
            len_base_nodes = len(base_nodes)
            long_cut -= 0.05

        vector = np.diff(points[base_nodes], axis=0)
        vector = vector / np.linalg.norm(vector, axis=1)[:, None]

        node_vector = np.zeros([len(base_nodes), 3])
        for i in range(len(base_nodes)):
            if i == 0:
                node_vector[i] = vector[0]
            elif i == len(base_nodes) - 1:
                node_vector[i] = vector[-1]
            else:
                node_vector[i] = (vector[i-1]+vector[i])/2

        node_vector = node_vector / np.linalg.norm(node_vector, axis=1)[:, None]

        angle = np.arccos(np.abs(np.dot(node_vector, sa_vector)))
        ind = np.where(angle > np.pi/3)[0][np.array([0,-1])]
        base_nodes = base_nodes[np.arange(ind[0], ind[1])]

        ctr.points = np.delete(ctr.points, base_nodes, axis=0)


def writeResults(fname, contours, frame=0):

    # Set up file writers.
    try:
        file = open(fname, "w", newline = "", encoding = "utf-8")
    except Exception as e:
        print(e)
        exit()

    # x    y    z    contour type    slice    weight    time frame
    def writePoint(point, ctype, slicenum, weight = 1):
        writer.writerow(["{:0.6f}".format(point[0]), "{:0.6f}".format(point[1]), "{:0.6f}".format(point[2]),
                         ctype, "{:d}".format(slicenum + 1), "{:0.4f}".format(weight), "{:d}".format(frame)])

    def writeContour(ctr):
        points = ctr.points
        if len(points.shape) == 2:
            for k in range(0, len(points)):
                writePoint(points[k], ctr.get_cname(), ctr.slice, weight=ctr.weight)
        else:
            writePoint(points, ctr.get_cname(), ctr.slice, ctr.weight)

    writer = csv.writer(file, delimiter = "\t")
    writer.writerow(["x", "y", "z", "contour type", "slice", "weight", "time frame"])

    for ctr in contours:
        writeContour(ctr)

    file.close()



def calculate_area_of_polygon_3d(points, normal):
    # Generate two orthonormal vectors to the normal
    v1 = np.array([1, 0, 0])
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    v1 = np.cross(v2, normal)
    v1 = v1 / np.linalg.norm(v1)

    # Define rotation matrix
    R = np.array([v1, v2, normal])

    # Rotate the points
    centroid = np.mean(points, axis=0)
    polygon = np.dot(points-centroid, R.T)[:, :2]

    # Compute the area of the polygon
    area = Polygon(polygon).area

    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    return area