#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/11/16 15:55:30

@author: Javiera Jilberto Vallejos 
'''
import numpy as np
import nibabel as nib

from segutils import getContoursFromMask, sharedRows, deleteHelper, getRVinsertIndices, getLAinsert, calculate_area_of_polygon_3d


class ViewSegFrame:
    def __init__(self, view, data, affine, zooms, labels, frame, translations=None):
        """
        Initialize the Seg2Contours class.

        Parameters:
        view (str): name of the view. Can be 'sa', 'la_2ch', 'la_3ch', or 'la_4ch'.
        data (np.array): 3D numpy array of the segmentation data.
        affine (np.array): 4x4 affine transformation matrix.
        zooms (list): pixel dimensions
        frame (int): frame number
        translations (optional, any): The translations parameter, default is None.
        """
        self.view = view
        self.data = data        
        self.affine = affine
        self.zooms = zooms
        self.labels = labels
        self.frame = frame

        self.contours = None
        self.z_normal = self.calculate_z_normal(affine)
        self.translations = translations


    @staticmethod
    def calculate_z_normal(affine):
        arr = np.array([[0,0,0],[0,0,1]])
        points = nib.affines.apply_affine(affine, arr)
        normal = points[1] - points[0]
        normal = normal/np.linalg.norm(normal)
        return normal


    def extract_slices(self, defseptum=False):
        slices = []
        for n in range(self.data.shape[2]):
            data_slice = self.data[:,:,n]
            if np.all(data_slice==0): continue   # avoid saving slices with no data

            # Calculate the origin of the slice
            origin = nib.affines.apply_affine(self.affine, np.array([0,0,n]))

            # Initialize slice
            slc = SegSlice(data_slice, origin, self.z_normal, n, 
                           self.affine, self.zooms[0], self.view, self.labels, 
                           defseptum=defseptum)
            if self.translations is not None:
                slc.accumulated_translation += self.translations[n]

            if slc.valid:
                slices += [slc]

        self.slices = slices
        return slices
    

class SegSlice:
    def __init__(self, data, origin, normal, slice_number, affine, pixdim, view, labels, lge_data=None, defseptum=False):
        self.data = data
        self.affine = affine
        self.origin = origin
        self.normal = normal
        self.slice_number = slice_number
        self.view = view
        self.labels = labels
        self.pixdim = pixdim

        self.defseptum = defseptum

        # getting pixel coords
        self.valid = self.get_boundaries(define_septum=defseptum)

        i = np.arange(data.shape[0])
        j = np.arange(data.shape[1])
        i, j = np.meshgrid(i,j)
        ij = np.vstack([i.flatten(),j.flatten()]).T
        self.ijk = np.hstack([ij, np.full((len(ij),1), slice_number)]).astype(float)

        # Getting bv pixel coordinates
        lv_ij = np.vstack(np.where(np.isclose(data, self.labels['lv']))).T
        self.lv_ijk = np.hstack([lv_ij, np.full((len(lv_ij),1), slice_number)]).astype(float)

        bv_ij = np.vstack(np.where(data>1.0)).T
        self.bv_ijk = np.hstack([bv_ij, np.full((len(bv_ij),1), slice_number)]).astype(float)

        all_ij = np.vstack(np.where(data>0.0)).T
        self.all_ijk = np.hstack([all_ij, np.full((len(all_ij),1), slice_number)]).astype(float)

        # Utils for optimization
        self.accumulated_translation = np.zeros(2)  # Only needed if aligning slices
        self.accumulated_matrix = np.eye(2)

        # If lge, we also store lge data
        if lge_data is not None:
            self.lge_data = lge_data

    def check_validity(self):
        if np.all(self.data == 0): return False
        return True
    
    def get_boundaries(self, define_septum=False):
        seg = self.data

        LVendo = np.isclose(seg, self.labels['lvbp'])
        LVepi = np.isclose(seg, self.labels['lv'])
        if not np.all(~LVepi):
            LVepi += LVendo
        RVendo = np.isclose(seg, self.labels['rv'])

        # Get contours
        LVendoCS = getContoursFromMask(LVendo, irregMaxSize = 20)
        LVepiCS = getContoursFromMask(LVepi, irregMaxSize = 20)
        RVendoCS = getContoursFromMask(RVendo, irregMaxSize = 20)

        is_2chr = False
        if (len(LVendoCS) == 0) and (len(LVepiCS) == 0) and (len(RVendoCS) > 0):    # 2CHr, only RV present
            is_2chr = True

        # Check that LVepi and LVendo do not share any points (in SA)
        if 'sa' in self.view:
            [dup, _, _] = sharedRows(LVepiCS, LVendoCS)
            if len(dup) > 0:  # If they share rows, the slice is not valid
                return False

        # Differentiate contours for RV free wall and RV septum.
        [RVseptCS, ia, ib] = sharedRows(LVepiCS, RVendoCS)
        RVFW_CS = deleteHelper(RVendoCS, ib, axis = 0) # Delete the rows whose indices are in ib.

        # Remove RV septum points from the LV epi contour.
        if define_septum:
            LVepiCS = deleteHelper(LVepiCS, ia, axis = 0)  # In LVepiCS, delete the rows with index ia.

        LVendoIsEmpty = LVendoCS is None or np.max(LVendoCS.shape) <= 2
        LVepiIsEmpty = LVepiCS is None or np.max(LVepiCS.shape) <= 2
        RVendoIsEmpty = RVendoCS is None or RVendoCS.size == 0

        if not RVendoIsEmpty:
            self.has_rv = True
        else:
            self.has_rv = False

        # If doing long axis, remove line segments at base which are common to LVendoCS and LVepiCS.
        if not LVendoIsEmpty:
            self.lvendo_ijk = np.hstack([LVendoCS, np.full((len(LVendoCS),1), self.slice_number)]).astype(float)
        else:
            self.lvendo_ijk = np.array([])
        if not LVepiIsEmpty:
            self.lvepi_ijk = np.hstack([LVepiCS, np.full((len(LVepiCS),1), self.slice_number)]).astype(float)
        else:
            self.lvepi_ijk = np.array([])
        if not RVendoIsEmpty:
            self.rvendo_ijk = np.hstack([RVFW_CS, np.full((len(RVFW_CS),1), self.slice_number)]).astype(float)
            if not is_2chr:
                if len(RVseptCS) > 0:
                    self.rvsep_ijk = np.hstack([RVseptCS, np.full((len(RVseptCS),1), self.slice_number)]).astype(float)
        else:
            self.rvendo_ijk = np.array([])

        if LVepiIsEmpty:
            print('WARNING: No LV epi segmentation in {}, slice {}'.format(self.view.upper(), (self.slice_number+1)))
        if LVendoIsEmpty:
            return False
        else:
            return True
    

    def get_xyz_trans(self, which, translation=np.zeros(2), use_cum_trans=True):
        # Get working ijk
        if which == 'lv':
            working_ijk = np.copy(self.lv_ijk)
        elif which == 'lvendo':
            working_ijk = np.copy(self.lvendo_ijk)
        elif which == 'lvepi':
            working_ijk = np.copy(self.lvepi_ijk)
        elif which == 'lvepisep':
            if self.has_rv:
                working_ijk = [np.copy(self.lvepi_ijk), np.copy(self.rvsep_ijk)]
                working_ijk = np.vstack(working_ijk)
            else:
                working_ijk = np.copy(self.lvepi_ijk)
        elif which == 'rvendo':
            if self.has_rv:
                working_ijk = np.copy(self.rvendo_ijk)
            else:
                return np.array([])
        elif which == 'rvsep':
            if self.has_rv:
                working_ijk = np.copy(self.rvsep_ijk)
            else:
                return np.array([])
        elif which == 'bv':
            working_ijk = np.copy(self.bv_ijk)
        elif which == 'all':
            working_ijk = np.copy(self.all_ijk)

        # Define translation
        if use_cum_trans:
            t = translation + self.accumulated_translation
        else:
            t = translation

        # apply translation
        working_ijk[:,0:2] += t

        return nib.affines.apply_affine(self.affine, working_ijk)


    def get_xyz_affine(self, which, affine=np.array([0.,0.,0.,0.,0.,0.]), use_cum_trans=True, use_cum_affine=True):
        """ affine is an array of length 6, first 4 are the matrix and the last two the translation"""
        # Get working ijk
        if which == 'lv':
            working_ijk = np.copy(self.lv_ijk)
        elif which == 'lvendo':
            working_ijk = np.copy(self.lvendo_ijk)
        elif which == 'lvepi':
            working_ijk = np.copy(self.lvepi_ijk)
        elif which == 'lvepisep':
            working_ijk = [np.copy(self.lvepi_ijk), np.copy(self.rvsep_ijk)]
            working_ijk = np.vstack(working_ijk)
        elif which == 'rvendo':
            working_ijk = np.copy(self.rvendo_ijk)
        elif which == 'bv':
            working_ijk = np.copy(self.bv_ijk)
        elif which == 'all':
            working_ijk = np.copy(self.all_ijk)
        elif which == 'contours':
            lvendo = np.copy(self.lvendo_ijk)
            lvepi = np.copy(self.lvepi_ijk)
            if self.has_rv:
                rv = np.copy(self.rvendo_ijk)
                working_ijk = np.vstack([lvendo,lvepi,rv])
            else:
                working_ijk = np.vstack([lvendo,lvepi])

        # Define translation
        if use_cum_trans:
            t = affine[4:] + self.accumulated_translation
        else:
            t = affine[4:]
        if use_cum_affine:
            M = (np.eye(2) + affine[0:4].reshape([2,2]))@self.accumulated_matrix
        else:
            M = np.eye(2) + affine[0:4].reshape([2,2])

        # Apply affine transform in centered coordinates to avoid weird deformations
        centroid = np.mean(working_ijk[:,0:2], axis=0)
        t += centroid - M@centroid

        # apply transform
        working_ijk[:,0:2] = working_ijk[:,0:2]@M.T + t

        return nib.affines.apply_affine(self.affine, working_ijk)


    def tocontours(self, downsample):
        contour_list = ['lvendo', 'lvepi']
        if self.has_rv:
            contour_list += ['rvsep', 'rvendo']

        contours = []
        contours_added = []
        for name in contour_list:
            try:
                contour_points = self.get_xyz_trans(which=name)
            except:
                continue
            contour_points = contour_points[0:(contour_points.size - 1):downsample, :]

            contour = SegSliceContour(contour_points, name, self.slice_number, self.view, self.normal)
            contours.append(contour)
            contours_added.append(name)

            if self.has_rv and name == 'rvendo':   # Save rv inserts
                tmpRV_insertIndices = getRVinsertIndices(contour_points)
                if len(tmpRV_insertIndices) == 0: continue
                rv_insert_points = contour_points[tmpRV_insertIndices]

                # For the LA we only want the apical point
                if ('la' in self.view):
                    rv_insert_points = getLAinsert(rv_insert_points, contours[0].points, contours[1].points)

                contour = SegSliceContour(rv_insert_points, 'rvinsert', self.slice_number, self.view, self.normal)
                contours.append(contour)

        return contours


class SegSliceContour:
    def __init__(self, points, ctype, slice_number, view, normal=None, weight=1):
        self.points = points
        self.slice = slice_number
        self.view = view
        self.ctype = ctype
        self.weight = weight
        self.normal = normal

    def get_cname(self):
        if self.ctype == 'rvinsert':
            return 'RV_INSERT'
        elif self.ctype == 'apexepi':
            return 'APEX_EPI_POINT'
        elif self.ctype == 'apexendo':
            return 'APEX_ENDO_POINT'
        elif self.ctype == 'rvapex':
            return 'APEX_RV_POINT'
        elif self.ctype == 'mv':
            return 'MITRAL_VALVE'
        elif self.ctype == 'tv':
            return 'TRICUSPID_VALVE'
        elif self.ctype == 'av':
            return 'AORTA_VALVE'

        # Get View
        if 'la' in self.view:
            name = 'LAX'
        elif 'sa' in self.view:
            name = 'SAX'
        else:
            raise "Unknown view"

        if self.ctype == 'lvendo':
            return name + '_LV_ENDOCARDIAL'
        elif self.ctype == 'lvepi':
            return name + '_LV_EPICARDIAL'
        elif self.ctype == 'rvendo':
            return name + '_RV_FREEWALL'
        elif self.ctype == 'rvsep':
            return name + '_RV_SEPTUM'
        elif self.ctype == 'rvinsert':
            return name + '_RV_INSERT'




def add_apex(contours, segs, adjust_weights=False):
    # ENDO: Split la and sa contours
    sa_contours = []
    la_contours = []
    mv_points = []
    for ctr in contours:
        if 'lvendo' in ctr.ctype:
            if 'la' in ctr.view:
                la_contours.append(ctr)
            else:
                sa_contours.append(ctr)
        elif 'mv' in ctr.ctype:
            mv_points.append(ctr.points)
    if len(la_contours) == 0:
        return
    mv_points = np.vstack(mv_points)

    # Need to find if the slices are ordered from apex to base or base to apex
    mv_centroid = np.mean(mv_points, axis=0)
    mv_ij = segs['sa'].inverse_transform(mv_centroid)
    nslices = segs['sa'].data.shape[-1]
    if mv_ij[2] > nslices/2:
        apex_slice1 = 0
        apex_slice2 = 1
    else:
        apex_slice1 = -1
        apex_slice2 = -2

    # Last SA contour is always the most apical one
    sa1_centroid = np.mean(sa_contours[apex_slice1].points, axis=0)
    sa2_centroid = np.mean(sa_contours[apex_slice2].points, axis=0)

    # Find long_axis vector
    la_vector = sa2_centroid - sa1_centroid
    la_vector = la_vector/np.linalg.norm(la_vector)

    # Find lowest point in each la contour
    aux = []
    for ctr in la_contours:
        dist = ctr.points@la_vector
        ind = np.argmin(dist)
        aux.append(np.append(ctr.points[ind], dist[ind]))
    aux = np.vstack(aux)
    points = aux[:,0:3]
    dist = aux[:,3]
    endo_la_apex = points[np.argmin(dist)]

    vector = sa1_centroid-endo_la_apex
    endo_apex = sa1_centroid - (la_vector*np.dot(vector, la_vector))

    ctr = SegSliceContour(endo_apex, 'apexendo', 0, 'la')
    contours += [ctr]


    # EPI: Split la and sa contours
    sa_contours = []
    la_contours = []
    for ctr in contours:
        if 'lvepi' in ctr.ctype:
            if 'la' in ctr.view:
                la_contours.append(ctr)
            else:
                sa_contours.append(ctr)

    if len(la_contours) == 0:
        return

    # Find lowest point in each la contour
    aux = []
    for ctr in la_contours:
        dist = ctr.points@la_vector
        ind = np.argmin(dist)
        aux.append(np.append(ctr.points[ind], dist[ind]))
    aux = np.vstack(aux)
    points = aux[:,0:3]
    dist = aux[:,3]
    epi_la_apex = points[np.argmin(dist)]

    dist_epi_endo = (endo_la_apex-epi_la_apex)@la_vector
    apex = endo_apex - (la_vector*dist_epi_endo)

    ctr = SegSliceContour(apex, 'apexepi', 0, 'la')
    contours += [ctr]


def add_rv_apex(contours, cmrs):
    # Split la and sa contours
    safw_contours = []
    lafw_contours = []
    sasep_contours = []
    lasep_contours = []
    lv_epi_contours = []
    tv_points = []
    lv_apex = []
    for ctr in contours:
        if 'rvendo' in ctr.ctype:
            if 'la' in ctr.view:
                lafw_contours.append(ctr)
            else:
                safw_contours.append(ctr)
        elif 'rvsep' in ctr.ctype:
            if 'la' in ctr.view:
                lasep_contours.append(ctr)
            else:
                sasep_contours.append(ctr)
        elif 'lvepi' in ctr.ctype:
            if 'la' in ctr.view:
                lv_epi_contours.append(ctr)
            else:
                lv_epi_contours.append(ctr)
        elif 'mv' in ctr.ctype:
            tv_points.append(ctr.points)
        elif 'apexendo' in ctr.ctype:
            lv_apex.append(ctr.points)
    if len(lafw_contours) == 0:
        return
    tv_points = np.vstack(tv_points)
    lv_apex = np.vstack(lv_apex)

    # Merge FW and SEP contours
    rv_contours = []
    for i in range(len(safw_contours)):
        if i > len(sasep_contours)-1:
            rv_contours.append(safw_contours[i])
        else:
            lim_point1 = sasep_contours[i].points[0]
            dist1 = np.linalg.norm(lim_point1-safw_contours[i].points, axis=1)
            insert1 = np.argmin(dist1)
            lim_point2 = sasep_contours[i].points[-1]
            dist2 = np.linalg.norm(lim_point2-safw_contours[i].points, axis=1)
            insert2 = np.argmin(dist2)
            if insert1 > insert2:
                inserts = insert2, insert1
                sep_points = sasep_contours[i].points[::-1]
            else:
                inserts = insert1, insert2
                sep_points = sasep_contours[i].points

            fw_points = np.vstack([safw_contours[i].points[inserts[1]:], safw_contours[i].points[0:inserts[0]+1]])
            points = np.vstack([sep_points, fw_points])
            rv_contours.append(SegSliceContour(points, 'rvendo', safw_contours[i].slice, safw_contours[i].view, safw_contours[i].normal))

    # Need to find if the slices are ordered from apex to base or base to apex
    tv_centroid = np.mean(tv_points, axis=0)
    tv_ij = cmrs['sa'].inverse_transform(tv_centroid)
    nslices = cmrs['sa'].data.shape[-1]
    if tv_ij[2] > nslices/2:
        apex_slice0 = 0
        apex_slice1 = 1
        apex_slice2 = 2
    else:
        apex_slice0 = -1
        apex_slice1 = -2
        apex_slice2 = -3


    # Last SA contour is always the most apical one
    sa0_centroid = np.mean(rv_contours[apex_slice0].points, axis=0)
    sa1_centroid = np.mean(rv_contours[apex_slice1].points, axis=0)
    sa2_centroid = np.mean(rv_contours[apex_slice2].points, axis=0)

    # Find the area of these slices
    sa1_area = calculate_area_of_polygon_3d(rv_contours[apex_slice1].points, normal=rv_contours[apex_slice1].normal)
    sa2_area = calculate_area_of_polygon_3d(rv_contours[apex_slice2].points, normal=rv_contours[apex_slice2].normal)

    # Long_axis vector
    la_vector = rv_contours[apex_slice1].normal
    # Check that la_vector points toward the base
    if np.dot(sa2_centroid-sa1_centroid, la_vector) < 0:
        la_vector = -la_vector
    sa1_z = np.dot(sa1_centroid-lv_apex, la_vector)
    sa2_z = np.dot(sa2_centroid-lv_apex, la_vector)


    # Extrapolate to find when the area becomes 0
    rv_apex_z = sa1_z + (0 - sa1_area)/(sa2_area - sa1_area)*(sa2_z - sa1_z)
    dist_z = np.abs(rv_apex_z-sa1_z)

    sa_la_vector = sa2_centroid - sa1_centroid
    sa_la_vector = sa_la_vector/np.linalg.norm(sa_la_vector)
    aux = -dist_z/(np.dot(sa_la_vector, la_vector))
    rv_apex = sa0_centroid + aux*sa_la_vector/2

    ctr = SegSliceContour(rv_apex, 'rvapex', 0, 'la')
    contours += [ctr]


