import os
import numpy as np
import pandas as pd
import functools
import time
import copy
import scipy.optimize
from cvxopt import matrix, solvers
from plotly import graph_objects as go
from scipy.spatial import cKDTree
import meshio as io
import trimesh

#local imports
from .surface_enum import Surface, ContourType, SURFACE_CONTOUR_MAP
from  .build_model_tools import basis_function_bspline, der_basis_function_bspline, der2_basis_function_bspline
from  .build_model_tools import generate_gauss_points, adjust_boundary_weights

##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz

import pathlib
bvpath = pathlib.Path(__file__).parent.resolve()
template_path = pathlib.Path.joinpath(bvpath, 'template')

class BiventricularModel():
    """ This class creates a surface from the control mesh, based on
    Catmull-Clark subdivision surface method. Surfaces have the following properties:

    Attributes:
       numNodes = 388                       Number of control nodes.
       numElements = 187                    Number of elements.
       numSurfaceNodes = 5810               Number of nodes after subdivision
                                            (surface points).
       control_mesh                         Array of x,y,z coordinates of
                                            control mesh (388x3).
       et_vertex_xi                         local xi position (xi1,xi2,xi3)
                                            for each vertex (5810x3).

       et_pos                               Array of x,y,z coordinates for each
                                            surface nodes (5810x3).
       et_vertex_element_num                Element num for each surface
                                            nodes (5810x1).
       et_indices                           Elements connectivities (n1,n2,n3)
                                            for each face (11760x3).
       basis_matrix                         Matrix (5810x388) containing basis
                                            functions used to evaluate surface
                                            at surface point locations
       matrix                               Subdivision matrix (388x5810).


       GTSTSG_x, GTSTSG_y, GTSTSG_z         Regularization/Smoothing matrices
                                            (388x388) along
                                            Xi1 (circumferential),
                                            Xi2 (longitudinal) and
                                            Xi3 (transmural) directions


       apex_index                           Vertex index of the apex

       et_vertex_start_end                  Surface index limits for vertices
                                            et_pos. Surfaces are sorted in
                                            the following order:
                                            LV_ENDOCARDIAL, RV septum, RV free wall,
                                            epicardium, mitral valve, aorta,
                                            tricuspid, pulmonary valve,
                                            RV insert.
                                            Valve centroids are always the last
                                            vertex of the corresponding surface

       surface_start_end                    Surface index limits for embedded
                                            triangles et_indices.
                                            Surfaces are sorted in the following
                                            order:  LV_ENDOCARDIAL, RV septum, RV free wall,
                                            epicardium, mitral valve, aorta,
                                            tricuspid, pulmonary valve, RV insert.

       mBder_dx, mBder_dy, mBder_dz         Matrices (5049x338) containing
                                            weights used to calculate gradients
                                            of the displacement field at Gauss
                                            point locations.

       Jac11, Jac12, Jac13                  Matrices (11968x388) containing
                                            weights used to calculate Jacobians
                                            at Gauss point location (11968x338).
                                            Each matrix element is a linear
                                            combination of the 388 control points.
                                            J11 contains the weight used to
                                            compute the derivatives along Xi1,
                                            J12 along Xi2 and J13 along Xi3.
                                            Jacobian determinant is
                                            calculated/checked on 11968 locations.
       fraction                 gives the level of the patch
                                (level 0 = 1,level 1 = 0.5,level 2 = 0.25)
       b_spline                  gives the 32 control points which need to be weighted
                                (for each vertex)
       patch_coordinates        patch coordinates
       boundary                 boundary
       phantom_points           Some elements only have an epi surface.
                                The phantomt points are 'fake' points on
                                the endo surface.


    """

    numNodes = 388
    numElements = 187
    numSurfaceNodes = 5810

    def __init__(self, control_mesh_dir, filemod="", label = 'default', build_mode = False):
        """ Return a Surface object whose control mesh is *control_mesh* and should be
            fitted to the dataset *DataSet*
            control_mesh is always the same - this is the RVLV template. If you change
            the template, you need to regenerate all the matrices.
            the build_mode allows to load the data needed to build smoothing matices
            if build_mode is set to False only data needed for fitting is loaded
        """
        self.build_mode = build_mode
        if not os.path.exists(control_mesh_dir):
            ValueError('Invalid directory name')

        self.label = label
        model_file = os.path.join(control_mesh_dir,"model" + filemod + ".txt")
        if not os.path.exists(model_file):
            ValueError('Missing model.txt file')
        self.control_mesh = (pd.read_table
                             (model_file, sep='\s+', header=None)).values
        if self.build_mode:
            et_vertex_xi_file = os.path.join(control_mesh_dir,"etVertexXi.txt")
            if not os.path.exists(et_vertex_xi_file):
                ValueError('Missing etVertexXi.txt file')
            self.et_vertex_xi = (pd.read_table(
                et_vertex_xi_file, sep='\s+', header=None)).values

            fraction_file = os.path.join(control_mesh_dir, "fraction.txt")
            if not os.path.exists(fraction_file):
                ValueError('Missing fraction.txt file')
            self.fraction = (pd.read_table(
                fraction_file, sep='\s+', header=None)).values

            b_spline_file = os.path.join(control_mesh_dir, "control_points_patches.txt")
            if not os.path.exists(b_spline_file):
                ValueError('Missing control_points_patches.txt file')
            self.b_spline = (pd.read_table(
                b_spline_file, sep='\s+', header=None)).values.astype(int)-1

            boundary_file = os.path.join(control_mesh_dir,"boundary.txt")
            if not os.path.exists(boundary_file):
                ValueError('Missing boundary.txt file')
            self.boundary = (pd.read_table(
                boundary_file, sep='\s+', header=None)).values.astype(int)

            phantom_points_file = os.path.join(control_mesh_dir, "phantom_points.txt")
            if not os.path.exists(phantom_points_file):
                ValueError('Missing phantom_points.txt file')
            self.phantom_points = (pd.read_table(
                phantom_points_file, sep='\s+', header=None)).values.astype(float)
            self.phantom_points[:,:17] = self.phantom_points[:,:17].astype(int)-1

            patch_coordinates_file = os.path.join(control_mesh_dir,
                                                  "patch_coordinates.txt")
            if not os.path.exists(patch_coordinates_file):
                ValueError('Missing patch_coordinates.txt file')
            self.patch_coordinates = (pd.read_table(
                patch_coordinates_file, sep='\s+', header=None)).values

            local_matrix_file = os.path.join(control_mesh_dir,
                                             "local_matrix.txt")
            if not os.path.exists(local_matrix_file):
                ValueError('Missing local_matrix.txt file')
            self.local_matrix = (pd.read_table(
                local_matrix_file, sep='\s+', header=None)).values

        subdivision_matrix_file = os.path.join(control_mesh_dir,
                                               "subdivision_matrix" + filemod + ".txt")
        if not os.path.exists(subdivision_matrix_file):
            ValueError('Missing subdivision_matrix.txt')

        self.matrix = (pd.read_table(subdivision_matrix_file,
                                     sep='\s+',
                                     header=None)).values.astype(float)

        self.et_pos = np.dot(self.matrix, self.control_mesh)

        et_index_file = os.path.join(control_mesh_dir,'ETIndicesSorted' + filemod + '.txt')
        if not os.path.exists(et_index_file):
            ValueError('Missing ETIndicesSorted.txt file')
        self.et_indices = (pd.read_table(et_index_file, sep='\s+',
                                            header=None)).values.astype(int)-1

        #et_index_thruWall_file = os.path.join(control_mesh_dir, 'ETIndicesThruWall.txt') #RB addition for MyoMass calc
        et_index_thruWall_file = os.path.join(control_mesh_dir, 'epi_to_septum_ETindices.txt')
        if not os.path.exists(et_index_thruWall_file):
            ValueError('Missing ETIndicesThruWall.txt file for myocardial mass calculation')
        self.et_indices_thruWall = (
            pd.read_table(et_index_thruWall_file, sep='\s+',
                          header=None)).values.astype(int)-1

        et_index_EpiLVRV_file = os.path.join(control_mesh_dir, 'ETIndicesEpiRVLV.txt') #RB addition for MyoMass calc
        if not os.path.exists(et_index_EpiLVRV_file):
            ValueError('Missing ETIndicesEpiRVLV.txt file for myocardial mass calculation')
        self.et_indices_EpiLVRV = (
            pd.read_table(et_index_EpiLVRV_file, sep='\s+',
                          header=None)).values.astype(int)-1


        GTSTSG_x_file = os.path.join(control_mesh_dir,'GTSTG_x.txt')
        if not os.path.exists(GTSTSG_x_file):
            ValueError(' Missing GTSTG_x.txt file')
        self.GTSTSG_x = (
            pd.read_table(GTSTSG_x_file, sep='\s+',
                          header=None)).values.astype(float)

        GTSTSG_y_file = os.path.join(control_mesh_dir,'GTSTG_y.txt')
        if not os.path.exists(GTSTSG_y_file):
            ValueError(' Missing GTSTG_y.txt file')
        self.GTSTSG_y = (
            pd.read_table(GTSTSG_y_file, sep='\s+',
                          header=None)).values.astype(float)

        GTSTSG_z_file = os.path.join(control_mesh_dir,'GTSTG_z.txt')
        if not os.path.exists(GTSTSG_z_file):
            ValueError(' Missing GTSTG_z.txt file')
        self.GTSTSG_z = (
            pd.read_table(GTSTSG_z_file, sep='\s+',
                          header=None)).values.astype(float)

        etVertexElementNum_file = os.path.join(control_mesh_dir,
                                               'etVertexElementNum.txt')
        if not os.path.exists(etVertexElementNum_file):
            ValueError('Missing etVertexElementNum.txt file')
        self.et_vertex_element_num = \
            (pd.read_table(etVertexElementNum_file,
                           sep='\s+',header=None)).values.astype(
                int)-1

        mBder_x_file = os.path.join(control_mesh_dir,'mBder_x.txt')
        if not os.path.exists(mBder_x_file):
            ValueError('Missing mBder_x.file')
        self.mBder_dx = (
            pd.read_table(mBder_x_file, sep='\s+',
                          header=None)).values.astype(float)
        mBder_y_file = os.path.join(control_mesh_dir,'mBder_y.txt')
        if not os.path.exists(mBder_y_file):
            ValueError('Missing mBder_y.file')
        self.mBder_dy = (
            pd.read_table(mBder_y_file, sep='\s+',
                          header=None)).values.astype(float)

        mBder_z_file = os.path.join(control_mesh_dir,'mBder_z.txt')
        if not os.path.exists(mBder_z_file):
            ValueError('Missing mBder_z.file')
        self.mBder_dz = (
            pd.read_table(mBder_z_file, sep='\s+',
                          header=None)).values.astype(float)

        jac11_file = os.path.join(control_mesh_dir,'J11.txt')
        if not os.path.exists(jac11_file):
            ValueError('Missing J11.txt file')

        self.Jac11 = (pd.read_table(jac11_file, sep='\s+',
                                    header=None)).values.astype(float)

        jac12_file = os.path.join(control_mesh_dir, 'J12.txt')
        if not os.path.exists(jac12_file):
            ValueError('Missing J12.txt file')

        self.Jac12 = (pd.read_table(jac12_file, sep='\s+',
                                    header=None)).values.astype(float)
        jac13_file = os.path.join(control_mesh_dir, 'J13.txt')
        if not os.path.exists(jac13_file):
            ValueError('Missing J13.txt file')

        self.Jac13 = (pd.read_table(jac13_file, sep='\s+',
                                    header=None)).values.astype(float)

        basic_matrix_file = os.path.join(control_mesh_dir,'basis_matrix.txt')
        if not os.path.exists(basic_matrix_file):
            ValueError('Missing basis_matrix.txt file')
        self.basis_matrix = (
            pd.read_table(basic_matrix_file,
                          sep='\s+',header=None)).values.astype(
            float)  # OK


        if filemod == '_mod':
            self.apex_endo_index = 38 #50# endo #5485 #epi
            self.apex_epi_index = 4343
        else:
            self.apex_endo_index = 29 #50# endo #5485 #epi
            self.apex_epi_index = 5485 #50# endo #5485 #epi

        self.et_vertex_start_end = np.array(
            [[0, 1499], [1500, 2164], [2165, 3223], [3224, 5581],
             [5582, 5630], [5631, 5655], [5656, 5696], [5697, 5729],
             [5730, 5809]])

        surface_label_file = os.path.join(control_mesh_dir,'surface_region' + filemod + '.txt')
        self.surfs = np.loadtxt(surface_label_file, dtype=int)

        labels=np.unique(self.surfs)
        start_end = []
        for i in labels:
            ind = np.where(self.surfs==i)[0]
            start_end.append([ind[0], ind[-1]])
        self.surface_start_end = np.array(start_end)


        valve_label_file = os.path.join(control_mesh_dir,'valve_elems' + filemod + '.txt')
        self.valve_elems = np.loadtxt(valve_label_file, dtype = int)

    def get_nodes(self):
        return self.et_pos
    def get_control_mesh_nodes(self):
        return self.control_mesh
    def get_surface_vertex_start_end_index(self, surface_name):
        """ Get Surface Start and End
                Input:
                    surface_name: surface name
                Output:
                    array containing first and last vertices index belonging to surface_name
        """
        #todo the surface start and end index needs a beter definition

        if surface_name == Surface.LV_ENDOCARDIAL:
            return self.et_vertex_start_end[0, :]

        if surface_name == Surface.RV_SEPTUM:
            return self.et_vertex_start_end[1, :]

        if surface_name == Surface.RV_FREEWALL:
            return self.et_vertex_start_end[2, :]

        if surface_name == Surface.EPICARDIAL:
            return self.et_vertex_start_end[3, :]

        if surface_name == Surface.MITRAL_VALVE:
            return self.et_vertex_start_end[4, :]

        if surface_name == Surface.AORTA_VALVE:
            return self.et_vertex_start_end[5, :]

        if surface_name == Surface.TRICUSPID_VALVE:
            return self.et_vertex_start_end[6, :]

        if surface_name == Surface.PULMONARY_VALVE:
            return self.et_vertex_start_end[7, :]

        if surface_name == Surface.RV_INSERT:
            return self.et_vertex_start_end[8, :]
        if surface_name == Surface.APEX_ENDO:
            return [self.apex_endo_index]*2
        if surface_name == Surface.APEX_EPI:
            return [self.apex_epi_index]*2

    def get_surface_start_end_index(self, surface_name):
        """ Get Surface Start and End
                Input:
                    surface_name: surface name
                Output:
                    array containing first and last vertices index belonging to surface_name
        """
        #todo the surface start and end index needs a beter definition

        if surface_name == Surface.LV_ENDOCARDIAL:
            return self.surface_start_end[0, :]

        if surface_name == Surface.RV_SEPTUM:
            return self.surface_start_end[1, :]

        if surface_name == Surface.RV_FREEWALL:
            return self.surface_start_end[2, :]

        if surface_name == Surface.EPICARDIAL:
            return self.surface_start_end[3, :]

        if surface_name == Surface.MITRAL_VALVE:
            return self.surface_start_end[4, :]

        if surface_name == Surface.AORTA_VALVE:
            return self.surface_start_end[5, :]

        if surface_name == Surface.TRICUSPID_VALVE:
            return self.surface_start_end[6, :]

        if surface_name == Surface.PULMONARY_VALVE:
            return self.surface_start_end[7, :]

    def is_diffeomorphic(self, new_control_mesh, min_jacobian):
        """ This function checks the Jacobian value at Gauss point location
            (I am using 3x3x3 per element).
            This function returns 0 if one of the determinants is below a given
            threshold and 1 otherwise.
            I usually use 0.1 to make sure that there is no intersection/folding
            (We can also use 0, but it might still give a positive jacobian
            if there are small intersections due to numerical approximation.
            Input:
                new_control_mesh: control mesh we want to check
            Output:
                min_jacobian: Jacobian threshold
            """

        boolean = 1
        for i in range(len(self.Jac11)):
            jacobi = np.array(
                [[np.inner(self.Jac11[i, :], new_control_mesh[:, 0]),
                  np.inner(self.Jac12[i, :], new_control_mesh[:, 0]),
                  np.inner(self.Jac13[i, :], new_control_mesh[:, 0])],
                 [np.inner(self.Jac11[i, :], new_control_mesh[:, 1]),
                  np.inner(self.Jac12[i, :], new_control_mesh[:, 1]),
                  np.inner(self.Jac13[i, :], new_control_mesh[:, 1])],
                 [np.inner(self.Jac11[i, :], new_control_mesh[:, 2]),
                  np.inner(self.Jac12[i, :], new_control_mesh[:, 2]),
                  np.inner(self.Jac13[i, :], new_control_mesh[:, 2])]])
            determinant = np.linalg.det(jacobi)

            if determinant < min_jacobian:
                boolean = 0
                return boolean

        return boolean

    def CreateNextModel(self, DataSetES, ESTranslation):
        """Copy of the current model onto the next time model.
        Just the dataset is changed.
            Input:
                DataSetES: dataset for the new time frame
                ESTranslation: 2D translations needed
            Output:
                ESSurface: Copy of the current model ('self'), associated with the new DataSet DataSet.
        """

        ESSurface = copy.deepcopy(self)
        ESSurface.data_set = copy.deepcopy(DataSetES)
        ESSurface.SliceShiftES(ESTranslation, self.image_position_patient)

        return ESSurface

    def update_pose_and_scale(self,dataset):
        """ A method that initializes the model. It takes a DataSet and updates
        the model pose and scale
                 in accordance with the data points.

                 Input:
                     None
                 Output:
                     scaleFactor: scale factor between template and data points.
             """

        scale_factor = self._get_scaling(dataset)
        self.update_control_mesh(self.control_mesh * scale_factor)

        #todo: maybe is easier to define first the origin and the translation
        # to the origin and then rotation around origin


        # The rotation is defined about the origin so we need to translate the model to the origin
        self.update_control_mesh(self.control_mesh - self.et_pos.mean(axis=0))
        rotation = self._get_rotation(dataset)
        self.update_control_mesh( np.array([np.dot(rotation, node) for node in
                                      self.control_mesh]))

        # Translate the model back to origin of the DataSet coordinate system
        translation = self._get_translation(dataset)
        self.update_control_mesh(self.control_mesh+translation)
        rotation = self._get_rotation(dataset)
 # et_pos update

        return scale_factor

    def _get_scaling(self, dataset):
        '''
          Calculates a scaling factor for the model
          that rescale model to the reference shape
          Args:
            data_set(n x 3 NumPy array) an array containing x coodrinates of shape
            points as first column, y coords as second column and z coords as third
            column
           Returns:
            translation([x,y]) a NumPy array with x, y and z translationcoordinates
          '''
        model_shape_index = [self.apex_endo_index,
                       self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],
                       self.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1]]
        model_shape = np.array( self.et_pos[model_shape_index,:])
        reference_shape = np.array([dataset.apex_endo,
                           dataset.mitral_centroid,
                           dataset.tricuspid_centroid])
        mean_m = model_shape.mean(axis = 0)
        mean_r = reference_shape.mean(axis = 0)
        model_shape = model_shape -mean_m
        reference_shape = reference_shape - mean_r
        ss_model = (model_shape**2).sum()
        ss_reference = (reference_shape**2).sum()
        #centered Forbidius norm
        norm_model = np.sqrt(ss_model)
        reference_norm = np.sqrt(ss_reference)


        scaleFactor = reference_norm/norm_model

        return scaleFactor

    def _get_translation(self, dataset):
        '''
          Calculates a translation for (x, y, z)
          axis that centers data around the origin of the RV
          Args:
            data_set(n x 3 NumPy array) an array containing x coodrinates of shape
            points as first column, y coords as second column and z coords as third
            column
           Returns:
            translation([x,y]) a NumPy array with x, y and z translationcoordinates
          '''
        t_points_index_1 =  (dataset.contour_type ==
                            ContourType.SAX_RV_FREEWALL) | (
                         dataset.contour_type == ContourType.SAX_RV_FREEWALL) | (
                         dataset.contour_type == ContourType.PULMONARY_VALVE) | (
                         dataset.contour_type == ContourType.TRICUSPID_VALVE)
        t_points_index_2 = (dataset.contour_type ==ContourType.SAX_LV_ENDOCARDIAL) | \
                           (dataset.contour_type == ContourType.SAX_LV_ENDOCARDIAL) | (
                           dataset.contour_type == ContourType.MITRAL_VALVE) | (
                           dataset.contour_type == ContourType.AORTA_VALVE)
        # t_points_index = np.logical_or (dataset.contour_type == ContourType.SAX_RV_EPICARDIAL ,
        #                                 dataset.contour_type == ContourType.SAX_RV_SEPTUM)
        points_coordinates_1 = dataset.points_coordinates[t_points_index_1]
        points_coordinates_2 = dataset.points_coordinates[t_points_index_2]
        translation = (points_coordinates_1.mean(
            axis=0)+points_coordinates_2.mean(axis=0))*0.5
        return translation

    def  _get_rotation(self, data_set):

        # computes the rotation between model and data set
        # the rotation is given by considering the x -axis direction defined
        # by the mitral valve centroid and apex
        # the origin of the coordinates system is the mid point between the
        # apex and mitral centroid

        base = data_set.mitral_centroid

        # computes data_set coordinates system
        xaxis = data_set.apex_endo - base
        xaxis = xaxis / np.linalg.norm(xaxis)

        apex_position_model = self.et_pos[self.apex_endo_index, :]
        base_model = self.et_pos[
                     self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1], :]

        xaxis_model = apex_position_model- base_model
        xaxis_model = xaxis_model / np.linalg.norm(xaxis_model) #normalize

        # compute origin defined at 1/3 of the height of the model on the Ox
        # axis
        tempOrigin = 0.5 * (data_set.apex_endo + base)
        tempOrigin_model = 0.5*(apex_position_model +base_model)

        maxd = np.linalg.norm(0.5 * (data_set.apex_endo - base))
        mind = -np.linalg.norm(0.5 * (data_set.apex_endo - base))

        maxd_model = np.linalg.norm(0.5 * (apex_position_model- base_model))
        mind_model = -np.linalg.norm(0.5 * (apex_position_model - base_model))

        point_proj = data_set.points_coordinates[(data_set.contour_type ==
                                                  ContourType.LAX_LV_ENDOCARDIAL), :]
        #point_proj = np.vstack((point_proj,data_set.points_coordinates[
        #                       (data_set.contour_type == ContourType.LAX_RV_FREEWALL), :]))
        #point_proj = np.vstack((point_proj,data_set.points_coordinates[
        #                       (data_set.contour_type == ContourType.SAX_RV_SEPTUM), :]))
        #point_proj = np.vstack((point_proj,data_set.points_coordinates[
        #                       (data_set.contour_type == ContourType.LAX_RV_SEPTUM), :]))

        #point_proj = np.vstack((point_proj,
        #                        data_set.points_coordinates[
        #                        ( data_set.contour_type == ContourType.SAX_RV_FREEWALL), :]))
        point_proj = np.vstack((point_proj,
                                data_set.points_coordinates[
                                ( data_set.contour_type == ContourType.LAX_LV_ENDOCARDIAL), :]))

        if len(point_proj) == 0:
            point_proj = np.vstack((point_proj,data_set.points_coordinates[
                                  (data_set.contour_type == ContourType.SAX_RV_SEPTUM), :]))
            point_proj = np.vstack((point_proj,
                                   data_set.points_coordinates[
                                   ( data_set.contour_type == ContourType.SAX_RV_FREEWALL), :]))

        if len(point_proj) == 0:
            ValueError('Missing contours in update_pose_and_scale')
            return


        tempd = [np.dot(xaxis, p) for p in (point_proj - tempOrigin)]
        maxd = max(np.max(tempd), maxd)
        mind = min(np.min(tempd), mind)

        model_epi = self.et_pos[self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[0]:
                               self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1]+ 1,:]
        tempd_model = [np.dot(xaxis_model, point_model)
                       for point_model in (model_epi- tempOrigin_model)]
        maxd_model = max(np.max(tempd_model), maxd_model)
        mind_model = min(np.min(tempd_model), mind_model)

        centroid = tempOrigin + mind * xaxis  + ((maxd - mind) / 3.0) * xaxis
        centroid_model = tempOrigin_model + mind_model * xaxis_model + \
                         ((maxd_model - mind_model) / 3.0) * xaxis_model


        #Compute Oy axis
        valid_index = (data_set.contour_type == ContourType.SAX_RV_FREEWALL) + \
                      (data_set.contour_type == ContourType.SAX_RV_SEPTUM)
        rv_endo_points = data_set.points_coordinates[ valid_index, :]

        rv_points_model = \
            self.et_pos[ self.get_surface_vertex_start_end_index(Surface.RV_SEPTUM)[0]:
                         self.get_surface_vertex_start_end_index(Surface.RV_FREEWALL)[1] + 1, :]

        rv_centroid = rv_endo_points.mean(axis=0)
        rv_centroid_model = rv_points_model.mean(axis=0)

        scale = np.dot(xaxis, rv_centroid) - np.dot(xaxis, centroid)/np.dot(xaxis,xaxis)
        scale_model = np.dot(xaxis_model, rv_centroid_model) - np.dot(xaxis_model,
                     centroid_model)/np.dot(xaxis_model,xaxis_model)
        rvproj = centroid + scale * xaxis
        rvproj_model = centroid_model + scale_model * xaxis_model


        yaxis = rv_centroid - rvproj
        yaxis_model = rv_centroid_model - rvproj_model

        yaxis = yaxis / np.linalg.norm(yaxis)
        yaxis_model = yaxis_model / np.linalg.norm(yaxis_model)

        zaxis = np.cross(xaxis, yaxis)
        zaxis_model = np.cross(xaxis_model, yaxis_model)

        zaxis = zaxis / np.linalg.norm(zaxis)
        zaxis_model = zaxis_model / np.linalg.norm(zaxis_model)

        # Find translation and rotation between the two coordinates systems
        """ The easiest way to solve it (in my opinion) is by using a
        Singular Value Decomposition as reported by Markley (1988):
            1. Obtain a matrix B as follows:
                B=∑ni=1aiwiviTB=∑i=wiviT
            2. Find the SVD of BB
                B=USVT
            3. The rotation matrix is:
                R=UMVT, where M=diag([11det(U)det(V)])
        """

        # Step 1
        B = np.outer(xaxis, xaxis_model) \
            + np.outer(yaxis, yaxis_model) \
            + np.outer(zaxis, zaxis_model)

        # Step 2
        [U, s, Vt] = np.linalg.svd(B)

        M = np.array([[1, 0, 0], [0, 1, 0],
                      [0, 0, np.linalg.det(U) * np.linalg.det(Vt)]])
        Rotation = np.dot(U, np.dot(M, Vt))

        return Rotation

    def generate_contraint_matrix(self):
        """ This function generates constraints matrix to be given to cvxopt
            Input:
                None
            Output:
                constraints: constraints matrix
            """

        constraints = []
        for i in range(len(self.mBder_dx)):  # rows and colums will always be the same so
            # we just need to precompute this and then calculate the values...

            dXdxi = np.zeros((3, 3), dtype='float')

            dXdxi[:, 0] = np.dot(self.mBder_dx[i, :], self.control_mesh)
            dXdxi[:, 1] = np.dot(self.mBder_dy[i, :], self.control_mesh)
            dXdxi[:, 2] = np.dot(self.mBder_dz[i, :], self.control_mesh)

            g = np.linalg.inv(dXdxi)

            Gx = np.dot(self.mBder_dx[i, :], g[0, 0]) + np.dot(
                self.mBder_dy[i, :], g[1, 0]) + np.dot(self.mBder_dz[i, :],
                                                       g[2, 0])
            constraints.append(Gx)

            Gy = np.dot(self.mBder_dx[i, :], g[0, 1]) + np.dot(
                self.mBder_dy[i, :], g[1, 1]) + np.dot(self.mBder_dz[i, :],
                                                       g[2, 1])
            constraints.append(Gy)

            Gz = np.dot(self.mBder_dx[i, :], g[0, 2]) + np.dot(
                self.mBder_dy[i, :], g[1, 2]) + np.dot(self.mBder_dz[i, :],
                                                       g[2, 2])
            constraints.append(Gz)

        return np.asmatrix(constraints)

    def MultiThreadSmoothingED(self, weight_GP, data_set):
        """ This function performs a series of LLS fits. At each iteration the
        least squares optimisation is performed and the determinant of the
        Jacobian matrix is calculated.
        If all the values are positive, the subdivision surface is deformed by
        updating its control points, projections are recalculated and the
        regularization weight is decreased.
        As long as the deformation is diffeomorphic, smoothing weight is decreased.
            Input:
                case: case name
                weight_GP: data_points' weight
            Output:
                None. 'self' is updated in the function itself
        """
        start_time = time.time()
        high_weight = weight_GP*1E+10  # First regularization weight
        isdiffeo = 1
        iteration = 1
        factor = 5
        min_jacobian = 0.1

        while (isdiffeo == 1) & (high_weight > weight_GP*1e2) & (iteration <50):

            displacement, err  = self.lls_fit_model(weight_GP, data_set,
                                                    high_weight)
            print('     Iteration #' + str(iteration) + ' ICF error ' + str(err))

            isdiffeo = self.is_diffeomorphic(np.add(self.control_mesh, displacement),
                                             min_jacobian)

            if isdiffeo == 1:
                self.update_control_mesh(np.add(self.control_mesh, displacement))
                high_weight = high_weight / factor  # we divide weight by 'factor' and start again...


            else:
                # If Isdiffeo ==1, the model is not updated.
                # We divide factor by 2 and try again.

                if  factor > 1:

                    factor = factor / 2
                    high_weight = high_weight * factor
                    isdiffeo = 1



            iteration = iteration + 1

        print("End of the implicitly constrained fit")
        print("--- %s seconds ---" % (time.time() - start_time))
        return high_weight


    def lls_fit_model(self, weight_GP, data_set, smoothing_Factor ):
        zfactor = 0.001
        [index, weights, distance_prior, projected_points_basis_coeff] = \
            self.compute_data_xi(weight_GP, data_set)

        prior_position = np.linalg.multi_dot([projected_points_basis_coeff, self.control_mesh])

        w = weights * np.identity(len(prior_position))
        WPG = np.linalg.multi_dot([w, projected_points_basis_coeff])

        GTPTWTWPG = np.linalg.multi_dot([WPG.T, WPG])
        # np.linalg.multi_dot faster than np.dot
        A = GTPTWTWPG + smoothing_Factor * (
                self.GTSTSG_x + self.GTSTSG_y + zfactor * self.GTSTSG_z)
        data_points_position = data_set.points_coordinates[index]
        Wd = np.linalg.multi_dot([w, data_points_position - prior_position])
        rhs = np.linalg.multi_dot([WPG.T, Wd])

        solf = np.linalg.solve(A.T.dot(A), A.T.dot(rhs))  # solve the Moore-Penrose pseudo inversee

        err = np.linalg.norm( data_points_position - prior_position, axis =1)
        err = np.sum(np.power(err, 2))
        err = np.sqrt(err/len(prior_position))

        return  solf , err

    def FitModel(self, data_set, weight_GP,
                 low_smoothing_weight, transmural_weight):
        """ This function creates mitral, tricuspid and RV epicardial phantom
        points and calls MultiThreadSmoothingED and MultiThreadSmoothingDiffeoED.
        Mitral and tricuspid phantom points are created to force the valve to
        stay circular.
        RV epicardial phantom points are created to 'help' as the RV epicardium
         was not contoured in the UK Biobank (if your dataset contains RV
         epicardial points, you don't need to do it).
         If you have points on the aorta and pulmonary, you can create
            phantom points (and I recommend it).

            Input:
                case: case name
                time_frame: time frame (ED = 1)
                saving_path: path where models and contours are going to be saved.

            Output:
                None
        """

        mitral_valve = data_set.create_valve_phantom_points(30)
        data_set.points_coordinates = np.vstack(
            (data_set.points_coordinates, mitral_valve))
        data_set.slice_number = np.hstack(
            (data_set.slice_number, [-1] * len(mitral_valve)))
        data_set.contour_type = np.hstack(
            (data_set.contour_type, [ContourType.MITRAL_PHANTOM] * len(mitral_valve)))

        Tri_points = data_set.Create_Tricuspid_phantomPoints(30)
        data_set.points_coordinates = np.vstack(
            (data_set.points_coordinates, Tri_points))
        data_set.slice_number = np.hstack(
            (data_set.slice_number, [-1] * len(Tri_points)))
        data_set.contour_type = np.hstack(
            (data_set.contour_type, [ContourType.TRICUSPID_PHANTOM] * len(Tri_points)))

        RV_epi = data_set.create_rv_epicardium()
        data_set.points_coordinates = np.vstack(
            (data_set.points_coordinates, RV_epi[:, 0:3]))
        data_set.slice_number = np.hstack(
            (data_set.slice_number, RV_epi[:, 4])).data_set.contour_type = np.hstack((data_set.contour_type,
                                                                                      [ContourType.SAX_RV_EPICARDIAL] * len(
                                                                                          RV_epi[:, 3])))

        # Circle does not separate LAX RV contours into RV_SEPTUM and RV_FREEWALL
        # so we need to do it ourselves. Again, you can comment it if you don't
        # have this issue.
        data_set.Identify_RVS_LAX()

        self.MultiThreadSmoothingED(data_set, weight_GP)
        self.SolveProblemCVXOPT(data_set, weight_GP, low_smoothing_weight,
                                transmural_weight)

    def SolveProblemCVXOPT(self, data_set, weight_GP, low_smoothing_weight,
                           transmural_weight):
        """ This function performs the proper diffeomorphic fit.
            Input:
                case: case name
                weight_GP: data_points' weight
                low_smoothing_weight: smoothing weight (for regularization term)
            Output:
                None. 'self' is updated in the function itself
        """
        start_time = time.time()
        [ data_points_index, w_out, distance_prior ,projected_points_basis_coeff] = \
            self.compute_data_xi(weight_GP,  data_set)

        # exlude the outliers defined as (u-mean(u)) > 6*std(u)
        projected_points_basis_coeff = projected_points_basis_coeff[abs(
            distance_prior - np.mean(distance_prior)) < 6 * np.std(distance_prior), :]
        data_points_index = data_points_index[
            abs(distance_prior - np.mean(distance_prior)) < 6 * np.std(distance_prior)]
        w_out = w_out[
            abs(distance_prior - np.mean(distance_prior)) < 6 * np.std(distance_prior)]

        data_points = data_set.points_coordinates[data_points_index]

        prior_position = np.dot(projected_points_basis_coeff, self.control_mesh)
        w = w_out * np.identity(len(prior_position))
        WPG = np.dot(w, projected_points_basis_coeff)
        GTPTWTWPG = np.dot(WPG.T, WPG)

        A = GTPTWTWPG + low_smoothing_weight * (
                self.GTSTSG_x + self.GTSTSG_y + transmural_weight * self.GTSTSG_z)
        Wd = np.dot(w, data_points - prior_position)
        # rhs = np.dot(WPG.T, Wd)

        previous_step_err = 0
        tol = 1e-6
        iteration = 0

        Q = 2 * A  # .T*A  # 2*A
        quadratic_form = matrix(0.5 * (Q + Q.T),
                                tc='d')  # to make it symmetrical.
        prev_displacement = np.zeros((self.numNodes, 3))

        step_err = np.linalg.norm(data_points - prior_position, axis=1)
        step_err = np.sum(np.power( step_err, 2))
        step_err = np.sqrt(step_err/len(prior_position))
        print('Explicitly constrained fit')
        while abs(step_err - previous_step_err) > tol and iteration < 10:
            print('     Iteration #' + str(iteration + 1) + ' ECF error ' + str(
                step_err))
            previous_step_err = step_err

            linear_part_x = matrix((2 * np.dot(prev_displacement[:, 0].T, A)
                                    - 2 * np.dot(Wd[:, 0].T,WPG).T), tc='d')
            linear_part_y = matrix((2 * np.dot(prev_displacement[:, 1].T, A)
                                    - 2 * np.dot(Wd[:, 1].T, WPG).T), tc='d')
            linear_part_z = matrix((2 * np.dot(prev_displacement[:, 2].T,A)
                                    - 2 * np.dot(Wd[:, 2].T,WPG).T), tc='d')

            linConstraints = matrix(self.generate_contraint_matrix(), tc='d')
            linConstraintNeg = -linConstraints

            G = matrix(np.vstack((linConstraints, linConstraintNeg)))
            size = 2 * (3 * len(self.mBder_dx))
            bound = 1 / 3
            h = matrix([bound] * size)

            solvers.options['show_progress'] = False

            #  Solver: solvers.qp(P,q,G,h)
            #  see https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
            #  for explanation
            solx = solvers.qp(quadratic_form, linear_part_x, G, h)
            soly = solvers.qp(quadratic_form, linear_part_y, G, h)
            solz = solvers.qp(quadratic_form, linear_part_z, G, h)

            sx = []
            sy = []
            sz = []

            for a in solx['x']:
                sx.append(a)

            for a in soly['x']:
                sy.append(a)

            for a in solz['x']:
                sz.append(a)

            displacement = np.zeros((self.numNodes, 3))

            displacement[:, 0] = np.asarray(sx)
            displacement[:, 1] = np.asarray(sy)
            displacement[:, 2] = np.asarray(sz)

            # check if diffeomorphic
            Isdiffeo = self.is_diffeomorphic(  np.add(self.control_mesh,
                                                      displacement), 0.1)

            if Isdiffeo == 0:
                # Due to numerical approximations, epicardium and endocardium
                # can 'touch' (but not cross),
                # leading to a negative jacobian. If it happens, we stop.
                prev_displacement[:, 0] = prev_displacement[:, 0] + 0.1*displacement[:, 0]
                prev_displacement[:, 1] = prev_displacement[:, 1] + 0.1*displacement[:, 1]
                prev_displacement[:, 2] = prev_displacement[:, 2] + 0.1*displacement[:, 2]

                self.update_control_mesh(self.control_mesh + displacement)
                prior_position = np.dot(projected_points_basis_coeff,
                                        self.control_mesh)
                step_err = np.linalg.norm(data_points - prior_position, axis=1)
                step_err = np.sum(np.power(step_err, 2))
                step_err = np.sqrt(step_err / len(prior_position))
                iteration = iteration + 1
                print('Diffeomorphic condition not verified ')
                break

            else:
                prev_displacement[:, 0] = prev_displacement[:, 0] + sx
                prev_displacement[:, 1] = prev_displacement[:, 1] + sy
                prev_displacement[:, 2] = prev_displacement[:, 2] + sz

                self.update_control_mesh(self.control_mesh + displacement)

                prior_position = np.dot(projected_points_basis_coeff,
                                        self.control_mesh)
                step_err = np.linalg.norm(data_points - prior_position, axis=1)
                step_err = np.sum(np.power(step_err, 2))
                step_err = np.sqrt(step_err / len(prior_position))
                iteration = iteration + 1



        print("--- End of the explicitly constrained fit ---")
        print("--- %s seconds ---" % (time.time() - start_time))



    def PlotSurface(self, face_color_LV, face_color_RV, face_color_epi, my_name,
                    surface="all", opacity=0.8):
        """ Plot 3D model.
            Input:
               face_color_LV, face_color_RV, face_color_epi: LV_ENDOCARDIAL, RV and epi colors
               my_name: surface name
               surface (optional): all = entire surface,
               endo = endocardium, epi = epicardium  (default = "all")
            Output:
               triangles_epi, triangles_LV, triangles_RV: triangles that
               need to be plotted for the epicardium, LV_ENDOCARDIAL and Rv respectively
               lines: lines that need to be plotted
        """

        x = np.array(self.et_pos[:, 0]).T
        y = np.array(self.et_pos[:, 1]).T
        z = np.array(self.et_pos[:, 2]).T

        # LV_ENDOCARDIAL endo
        surface_index = self.get_surface_start_end_index(Surface.LV_ENDOCARDIAL)
        I_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1, 0])
        J_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,1])
        K_LV = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1] + 1,2])
        simplices_lv = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # RV free wall
        surface_index = self.get_surface_start_end_index(Surface.RV_FREEWALL)
        I_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          0] )
        J_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          1] )
        K_FW = np.asarray(self.et_indices[
                          surface_index[0]:surface_index[1]+1,
                          2] )
        simplices_fw = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # RV septum
        surface_index = self.get_surface_start_end_index(Surface.RV_SEPTUM)
        I_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         0] )
        J_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         1] )
        K_S = np.asarray(self.et_indices[
                         surface_index[0]:surface_index[1]+1,
                         2] )
        simplices_s = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        # Epicardium
        surface_index = self.get_surface_start_end_index(Surface.EPICARDIAL)
        I_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 0] )
        J_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 1] )
        K_epi = np.asarray(self.et_indices[
                           surface_index[0]:surface_index[1]+1, 2] )
        simplices_epi = np.vstack(
            (self.et_indices[surface_index[0]:surface_index[1] + 1, 0],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 1],
             self.et_indices[surface_index[0]:surface_index[1] + 1, 2])).T

        if surface == "all":
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T

            tri_vertices_epi = list(map(lambda index: points3D[index], simplices_epi))
            tri_vertices_fw = list(
                map(lambda index: points3D[index], simplices_fw))
            tri_vertices_s = list(
                map(lambda index: points3D[index], simplices_s))
            tri_vertices_lv= list(
                map(lambda index: points3D[index], simplices_lv))

            triangles_LV = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_LV,
                i=I_LV,
                j=J_LV,
                k=K_LV,
                opacity=1,
                name = 'LV edocardial',
                showlegend = True,
            )

            triangles_FW = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV free wall',
                showlegend=True,
                i=I_FW,
                j=J_FW,
                k=K_FW,
                opacity=1
            )

            triangles_S = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV septum',
                showlegend=True,
                i=I_S,
                j=J_S,
                k=K_S,
                opacity=1
            )

            triangles_epi = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_epi,
                i=I_epi,
                j=J_epi,
                k=K_epi,
                opacity=0.4,
                name = 'epicardial',
                showlegend=True,
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_epi]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_epi = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe epicardial'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_fw]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_fw = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe rv free wall'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_s]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_s = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe rv septum'
            )

            lists_coord = [
                [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices_lv]
                for c in range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines_lv = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,

                name = 'wireframe lv edocardial'
            )

            return [triangles_epi, triangles_LV, triangles_FW,triangles_S,
                    lines_epi, lines_lv,lines_fw,lines_s]

        if surface == "endo":
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T
            simplices = np.vstack((self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2, 1]+1, 0] ,
                                   self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2, 1]+1,1] ,
                                   self.et_indices[
                                   self.surface_start_end[0, 0]:
                                   self.surface_start_end[2,1]+1,2] )).T

            tri_vertices = list(map(lambda index: points3D[index], simplices))

            triangles_LV = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_LV,
                i=I_LV,
                j=J_LV,
                k=K_LV,
                opacity=opacity,
                name = 'LV edocardial',
                showlegend=True,
            )

            triangles_FW = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV freewall',
                showlegend=True,
                i=I_FW,
                j=J_FW,
                k=K_FW,
                opacity=opacity)

            triangles_S = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_RV,
                name='RV septum',
                showlegend=True,
                i=I_S,
                j=J_S,
                k=K_S,
                opacity=opacity)
            # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            lists_coord = [
                [[T[k % 3][c] for k in range(4)] for T in tri_vertices] for c in
                range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                showlegend=True,
                name = 'wireframe'
            )

            return [triangles_LV, triangles_FW, lines]

        if surface == "epi":
            surface_index = self.get_surface_start_end_index(Surface.EPICARDIAL)
            points3D = np.vstack(
                (self.et_pos[:, 0], self.et_pos[:, 1], self.et_pos[:, 2])).T
            simplices = np.vstack((self.et_indices[surface_index[ 0]:
                                                   surface_index[1]+1,0] ,
                                   self.et_indices[
                                   surface_index[ 0]:
                                   surface_index[ 1]+1,1] ,
                                   self.et_indices[surface_index[ 0]:
                                                   surface_index[ 1]+1,2] )).T

            tri_vertices = list(map(lambda index: points3D[index], simplices))

            triangles_epi = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=face_color_epi,
                i=I_epi,
                j=J_epi,
                k=K_epi,
                opacity=0.8,
                name = 'epicardial',
                showlegend=True
            )

            # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
            lists_coord = [
                [[T[k % 3][c] for k in range(4)] for T in tri_vertices] for c in
                range(3)]
            Xe, Ye, Ze = [functools.reduce(lambda x, y: x + y, lists_coord[k])
                          for k in range(3)]

            # define the lines to be plotted
            lines = go.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=go.scatter3d.Line(color='rgb(0,0,0)', width=1.5),
                name = 'wireframe',
                showlegend=True
            )

            return [triangles_epi, lines]

    def get_intersection_with_plane(self, P0, N0, surface_to_use = None):
        ''' Calculate intersection points between a plane with the
            biventricular model (LV_ENDOCARDIAL only)
            P = L.GetIntersectiontWithPlane(P0,N0,'opt1',val1,...)
            [P,Fidx] = L.GetIntersectionWithPlane(P0,N0,'opt1',val1,...)

            The plane is defined by the normal N0 and a point P0 on the plane.
            The outputs P & Fidx are cell array where each cell defines the
            intersection for each surface (see the 'surfaces' option).
            - P{i} are Nx3 coordinate points on surface i that intersect with the
              plane.
            - Fidx{i} are indices of the surface faces indicating triangles that
              intersect the plane.  '''

        # Adjust P0 & N0 into a column vector
        N0 = N0 / np.linalg.norm(N0)

        Fidx = []

        if surface_to_use is None:
            surface_to_use = [Surface.LV_ENDOCARDIAL]
        for surface in surface_to_use:  # We just want intersection LV_ENDOCARDIAL,
            # RVS. RVFW, epi
            # Get the faces
            faces = self.get_surface_faces(surface)

            # --- find triangles that intersect with plane: (P0,N0)
            # calculate sign distance of each vertices

            # set the origin of the model at P0
            centered_vertex = self.et_pos - [list(P0)]*len(self.et_pos)
            # projection on the normal
            dist = np.dot(N0, centered_vertex.T)

            sgnDist = np.sign(dist[faces])
            # search for triangles having the vertex on the both sides of the
            # frame plane => intersecting with the frame plane
            valid_index = [np.any(sgnDist[i] > 0) and np.any(sgnDist[i] < 0)
                           for i in range(len(sgnDist))]
            intersecting_face_idx = np.where(valid_index)[0]

            if len(intersecting_face_idx) < 0:
                return np.empty((0,3))


            # Find the intersection lines - find segments for each intersected
            # triangles that intersects the plane
            # see http://softsurfer.com/Archive/algorithm_0104/algorithm_0104B.htm

            # pivot points

            iPos = [x for x in intersecting_face_idx if np.sum(sgnDist[x] >0) == 1] #all
            # triangles with one vertex on the positive part
            iNeg = [x for x in intersecting_face_idx if np.sum(sgnDist[x] <0) == 1] # all
            # triangles with one vertex on the negative part
            p1 = []
            u = []


            for face_index in iPos:  # triangles where only one
                # point
                # on positive side
                # pivot points
                pivot_point_mask = sgnDist[face_index, :] > 0
                res =centered_vertex[faces[face_index, pivot_point_mask ], :][0]
                p1.append(list(res))
                # u vectors
                u = u + list( np.subtract(
                    centered_vertex[faces[face_index,
                                          np.invert(pivot_point_mask)] ,:],
                    [list(res)]*2))


            for face_index in iNeg:  # triangles where only one
                # point on negative side
                # pivot points
                pivot_point_mask = sgnDist[face_index, :] < 0
                res = centered_vertex[faces[face_index, pivot_point_mask] ,
                      :][0] # select the vertex on the negative side
                p1.append(res)
                # u vectors
                u = u + list(np.subtract(
                    centered_vertex[faces[face_index,
                    np.invert(pivot_point_mask)],:],
                    [list(res)]*2)
                )


            # calculate the intersection point on each triangle side
            u = np.asarray(u).T
            p1 = np.asarray(p1).T
            if len(p1) == 0:
                continue
            mat = np.zeros((3, 2 * p1.shape[1]))
            mat[0:3, 0::2] = p1
            mat[0:3, 1::2] = p1
            p1 = mat

            sI = - np.dot(N0.T, p1) / (np.dot(N0.T, u))
            factor_u = np.array([list(sI)]*3)
            pts = np.add(p1, np.multiply(factor_u, u)).T
            # add vertices that are on the surface
            Pon = centered_vertex[faces[sgnDist == 0], :]
            pts = np.vstack((pts, Pon))
            # #change points to the original position
            Fidx = Fidx + list(pts + [list(P0)]*len(pts))

        return Fidx

    def get_intersection_with_dicom_image(self, frame,surface =None):
        ''' Get the intersection contour points between the biventricular model with a DICOM image '''

        image_position = np.asarray(frame.position,
                                    dtype=float)
        image_orientation = np.asarray(frame.orientation,
                                       dtype=float)


        # get image position and the image vectors
        V1 = np.asarray(image_orientation[0:3], dtype=float)
        V2 = np.asarray(image_orientation[3:6], dtype=float)
        V3 = np.cross(V1, V2)


        # get intersection points
        P = self.get_intersection_with_plane(image_position, V3, surface_to_use=surface)

        return P

    def get_surface_faces(self, surface):
        ''' Get the faces definition for a surface triangulation'''

        surface_index = self.get_surface_start_end_index(surface)
        return self.et_indices[ surface_index[0]:surface_index[1] + 1, :]

    def compute_data_xi(self, weight, data):
        """ This function calculates the data basis function matrix.
        It projects the N data points onto the closest point in the
            corresponding model surface. If 2 data points are projected
             onto the same surface point, the closest one is kept.
                Input:
                    Weight: weight given to the N data points

                Output:
                    Psi_matrix: basis function matrix (Nx388)
                    index: closest points indices (Nx1)
                    d: data points (Nx3)
                    W: Weight on the data points (N*N). Higher weights are given
                    to RV_insert and valve points.
                    contour_type: 0 if this point belongs to the LV_ENDOCARDIAL, 1 for RV.
                    This is to get RV and LV_ENDOCARDIAL error if needed (Nx1)
                    distance_d_prior: distances to the closest points (Nx1)
        """

        data_points = np.array(data.points_coordinates)
        data_contour_type = np.array(data.contour_type)
        data_weights =  np.array(data.weights)

        psi_matrix = []
        w = []
        distance_d_prior = []
        index = []
        data_points_index = []

        basis_matrix = self.basis_matrix

        # add by A. Mira : a more compressed way of initializing the cKDTree

        for surface in Surface:
            # Trees initialization

            surface_index = self.get_surface_vertex_start_end_index(surface)
            tree_points = self.et_pos[surface_index[0]:surface_index[1]+1, :]
            if len(tree_points) == 0:
                continue
            surface_tree = scipy.spatial.cKDTree(tree_points)

            # loop over contours is faster, for the same contours we are using
            # the same tree, therefor the query operation can be done for all
            # points of the same contour: A. Mira 02/2020
            for contour in SURFACE_CONTOUR_MAP[surface.value]:
                contour_points_index = np.where(data_contour_type == contour)[0]
                contour_points = data_points[contour_points_index]
                weights_gp =  data_weights[contour_points_index]
                if len(contour_points) == 0:
                    continue

                # LV_ENDOCARDIAL endo (sa: short
                # axis,
                # la: long axis)
                if surface.value < 4:  # these are the surfaces
                    distance, vertex_index = surface_tree.query(
                        contour_points, k=1, p=2)
                    index_closest = [x + surface_index[0] for x in vertex_index]

                    for i_idx, vertex_index in enumerate(index_closest):
                        if vertex_index not in index:
                            index.append(int(vertex_index))
                            data_points_index.append(
                                contour_points_index[i_idx])
                            psi_matrix.append(
                                basis_matrix[int(vertex_index), :])
                            w.append(weight*weights_gp[i_idx])
                            distance_d_prior.append(distance[i_idx])

                        else:
                            old_idx = index.index(vertex_index)
                            distance_old = distance_d_prior[old_idx]
                            if distance[i_idx] < distance_old:
                                distance_d_prior[old_idx] = distance[i_idx]
                                data_points_index[old_idx] = \
                                    contour_points_index[i_idx]
                                w[old_idx] = weight*weights_gp[i_idx]


                else:
                    # If it is a valve, we virtually translate the data points
                    # (only the ones belonging to the same surface) so their centroid
                    # matches the template's valve centroid.
                    # So instead of calculating the minimum distance between the point
                    # p and the model points pm, we calculate the minimum distance between
                    # the point p+t and pm,
                    # where t is the translation needed to match both centroids
                    # This is to make sure that the data points are going to be
                    # projected all around the valve and not only on one side.
                    if surface.value < 8:  # these are the landmarks without apex

                        # and rv inserts
                        centroid_valve = self.et_pos[surface_index[1]]
                        centroid_GP_valve = contour_points.mean(axis=0)
                        translation_GP_model =  centroid_valve - centroid_GP_valve
                        translated_points = np.add(contour_points,
                                                   translation_GP_model)
                    else:  # rv_inserts  and apex don't
                        # need to be translated
                        translated_points = contour_points
                    if contour in  [ContourType.MITRAL_PHANTOM,ContourType.PULMONARY_PHANTOM,ContourType.AORTA_PHANTOM,
                                    ContourType.TRICUSPID_PHANTOM]:
                        surface_tree = scipy.spatial.cKDTree(translated_points)
                        tree_points = tree_points[:-1]
                        distance, vertex_index = surface_tree.query(tree_points
                                                                    , k=1, p=2)
                        index_closest = [x+surface_index[0] for x in range(len(
                            tree_points))]
                        weights_gp = [ weights_gp[x] for x in vertex_index]

                        contour_points_index = [contour_points_index[x] for x
                                                in vertex_index]

                    else:
                        distance, vertex_index = surface_tree.query(
                            translated_points, k=1, p=2)
                        index_closest = []
                        for x in vertex_index:
                            if (x + surface_index[0]) != surface_index[1]:
                                index_closest.append(x + surface_index[0])
                            else:
                                index_closest.append(x + surface_index[0] - 1)

                    #
                    index = index + index_closest
                    psi_matrix = psi_matrix + list(basis_matrix[
                                                   index_closest, :])


                    w = w + [( weight * x) for x in weights_gp]

                    distance_d_prior = distance_d_prior + list(distance)
                    data_points_index = data_points_index + list(
                        contour_points_index)

        return [np.asarray(data_points_index), np.asarray(w),
                np.asarray(distance_d_prior), np.asarray(psi_matrix)]


    def evaluate_gauss_point(self, s, t, u, elem_number,
                             displacement_s=0,
                             displacement_t=0,
                             displacement_u=0):
        '''
        # This function evaluates the basis functions at the coordinates(u, v, t)
        # (it is not a surface point) for the element elem_number.
        Args:
            s: local coordinate
            t: local coordinate
            u: local coordinate
            elem_number: element number
            displacement_s: for finite difference calculation only (D-affine reg)
            displacement_t: for finite difference calculation only (D-affine reg)
            displacement_u: for finite difference calculation only (D-affine reg)

        Returns:
            full_matrix_coefficient_points: rows = number of data points
                                            columns = basis functions
                                            size = number of data points x 16
            full_matrix_coefficient_der(i, j, k): basis fonction for ith data point,
                                            jth basis fn, kth derivatives
                                            size = number of  data  points x 16 x 5:
                                             du, dv, duv, duu, dvv,
                                             dw, dvw, duw, dww, dudvdw
            control_points: 16 control points B - spline


        '''
        if not self.build_mode:
            print('To compute the smoothing matrix the model should be '
                  'read with build_mode=True')
            return

        # Allocate memory
        params_per_element = 32
        pWeights = np.zeros((params_per_element)) # weights
        dWeights = np.zeros((params_per_element, 10)) # weights derivatives

        matrix_coefficient_Bspline_points = np.zeros(params_per_element)
        matrix_coefficient_Bspline_der = np.zeros((params_per_element, 10))
        full_matrix_coefficient_points = np.zeros(( self.numNodes))
        full_matrix_coefficient_der = np.zeros(( self.numNodes, 10))

        #Find projection onto epi and endo surfaces

        ps = np.zeros(2)
        pt =np.zeros(2)
        fract = np.zeros(2)
        boundary_value = np.zeros(2)

        index_verts= self.et_vertex_element_num[:,0] == elem_number
        # endo surface
        index_endo = self.et_vertex_xi[:,2] == 0
        element_verts_xi = self.et_vertex_xi[np.logical_and(index_endo, index_verts),:2]

        if len(element_verts_xi) > 0:

            elem_tree = cKDTree(element_verts_xi)
            ditance,closest_vertex_id = elem_tree.query([s,t])
            index_endo = np.where(np.logical_and(index_endo, index_verts))[0][
                closest_vertex_id]


            ps[0] = element_verts_xi[closest_vertex_id, 0] - \
                    self.fraction[index_endo] * \
                    self.patch_coordinates[index_endo, 0]
            pt[0] = element_verts_xi[closest_vertex_id, 1] - \
                    self.fraction[ index_endo] * \
                    self.patch_coordinates[index_endo,1]

            boundary_value[0] = self.boundary[index_endo]
            fract[0] = 1 / self.fraction[index_endo]

            # normalize s, t coordinates
            s_endo = (s + displacement_s - ps[0]) / self.fraction[
                index_endo]
            t_endo = (t + displacement_t - pt[0]) / self.fraction[
                index_endo]
            b_spline_endo = self.b_spline[index_endo, :]

        elif elem_number >166:
            index_phantom = self.phantom_points[:, 0] == elem_number
            elem_phantom_points = self.phantom_points[index_phantom, :]
            elem_vertex_xi = np.stack((elem_phantom_points[:, 21],
                                       elem_phantom_points[:, 22])).T

            elem_tree = cKDTree(elem_vertex_xi)
            ditance, closest_vertex_id = elem_tree.query([s, t])
            ps[0] = elem_phantom_points[closest_vertex_id, 21] - \
                    elem_phantom_points[closest_vertex_id, 24] * \
                    elem_phantom_points[closest_vertex_id, 18]
            pt[0] = elem_phantom_points[closest_vertex_id, 22] - \
                    elem_phantom_points[closest_vertex_id, 24] * \
                    elem_phantom_points[closest_vertex_id, 19]

            boundary_value[0] = elem_phantom_points[closest_vertex_id, 17]
            fract[0] = 1 / elem_phantom_points[closest_vertex_id, 24]
            s_endo = (s + displacement_s - ps[0]) / \
                     elem_phantom_points[closest_vertex_id, 24]
            t_endo = (t + displacement_t - pt[0]) / \
                     elem_phantom_points[closest_vertex_id, 24]
            b_spline_endo = elem_phantom_points[closest_vertex_id, 1:17].astype(int)


        else:
            ValueError('Somethings wrong, i have no idea why')



        # epi surface
        index_epi = self.et_vertex_xi[:,2] == 1
        element_verts_xi = self.et_vertex_xi[np.logical_and(index_epi, index_verts),:2]
        #element_franc = self.fraction[np.logical_and(index_epi, index_verts)]
        #element_patch_coords = self.patch_coordinates[np.logical_and(index_epi, index_verts),:2]
        #element_bspline = self.b_spline[np.logical_and(index_epi, index_verts),:]
        elem_tree = cKDTree(element_verts_xi)
        ditance,closest_vertex_id = elem_tree.query([s,t])
        index_epi = np.where(np.logical_and(index_epi, index_verts))[0][closest_vertex_id]

        ps[1] = element_verts_xi[closest_vertex_id, 0] - \
                self.fraction[index_epi] * \
                self.patch_coordinates[index_epi, 0]
        pt[1] = element_verts_xi[closest_vertex_id, 1] - \
                self.fraction[ index_epi] * \
                self.patch_coordinates[index_epi,1]

        boundary_value[1] = self.boundary[index_epi ]
        fract[1] = 1 / self.fraction[index_epi]
        # normalize s, t coordinates
        s_epi = (s + displacement_s - ps[1]) / self.fraction[index_epi]
        t_epi = (t + displacement_t - pt[1]) / self.fraction[index_epi]
        b_spline_epi = self.b_spline[index_epi,:]


        u1 = u + displacement_u
        # normalize s, t coordinates
        control_points = np.concatenate((b_spline_endo, b_spline_epi))
        if len(control_points)<32:
            print('stop')
        # Uniform B - Splines basis functions
        sWeights = np.zeros((4,2))
        tWeights = np.zeros((4,2))
        uWeights = np.zeros(2)
        sWeights[:, 0] = basis_function_bspline(s_endo)
        tWeights[:, 0] = basis_function_bspline(t_endo)
        sWeights[:, 1] = basis_function_bspline(s_epi)
        tWeights[:, 1] = basis_function_bspline(t_epi)

        uWeights[0] = 1 - u1 # linear interpolation
        uWeights[1] = u1    # linear interpolation
        # Derivatives of the B - Splines basis functions
        ds =np.zeros((4,2))
        dt = np.zeros((4,2))
        du = np.zeros(2)

        ds[:, 0] = der_basis_function_bspline(s_endo)
        dt[:, 0] = der_basis_function_bspline(t_endo)
        ds[:, 1] = der_basis_function_bspline(s_epi)
        dt[:, 1] = der_basis_function_bspline(t_epi)
        du[0] = -1
        du[1] = 1

        # Second derivatives of the B - Splines basis functions
        dss = np.zeros((4,2))
        dtt = np.zeros((4,2))
        dss[:, 0] = der2_basis_function_bspline(s_endo)
        dtt[:, 0] = der2_basis_function_bspline(t_endo)
        dss[:, 1] = der2_basis_function_bspline(s_epi)
        dtt[:, 1] = der2_basis_function_bspline(t_epi)
        # Adjust the boundaries
        sWeights[:, 0], tWeights[:, 0] = adjust_boundary_weights(
            boundary_value[0],
            sWeights[:, 0],  tWeights[:, 0])
        sWeights[:, 1], tWeights[:, 1] = adjust_boundary_weights(
            boundary_value[1],
            sWeights[:, 1], tWeights[:, 1])
        ds[:, 0], dt[:, 0] = adjust_boundary_weights(boundary_value[0],
                                                     ds[:, 0], dt[:, 0])
        ds[:, 1], dt[:,1] = adjust_boundary_weights(boundary_value[1],
                                                    ds[:, 1], dt[:, 1])
        dss[:, 0], dtt[:, 0] = adjust_boundary_weights(boundary_value[0],
                                                       dss[:, 0], dtt[:, 0])
        dss[:, 1], dtt[:, 1] = adjust_boundary_weights(boundary_value[1],
                                                       dss[:, 1], dtt[:, 1])
        # Weights of the 16 tensors B - spline basis functions and their derivatives
        for k in range(2):
            for i in range(4):
                for j in range(4):
                    index = 16 * k + 4 * i + j
                    pWeights[index ] =  sWeights[j ,k] * tWeights[i ,k] * uWeights[ k ]

                    dWeights[index, 0] = ds[j, k ] * tWeights[i , k ] \
                                         * fract[k] * uWeights[k]

                    # dScale; % dphi / du = 2 ^ (p * n) * dx, where
                    #  n = level of the patch(0, 1 or 2) and p = order of differentiation.Here
                    #  p = 1 and n = 1 / self.fraction(indx)
                    dWeights[index, 1] = sWeights[j, k] * dt[i,k] * fract[k] * uWeights[k]

                    dWeights[index, 2] = ds[j, k] * dt[i,k] * \
                                         (fract[k] ** 2) * uWeights[k ]

                    dWeights[index, 3] = dss[j, k] * tWeights[i,k] *  (fract[k] ** 2) * uWeights[k ]

                    dWeights[index, 4] = sWeights[j, k] * dtt[i,k] * \
                                         (fract[k] ** 2) * uWeights[k]

                    dWeights[index, 5] = sWeights[j, k] * tWeights[i,k] * du[k]

                    dWeights[index, 6] = sWeights[j, k] * dt[i,k] * du[k] * fract[k ]

                    dWeights[index, 7] = ds[j, k] * tWeights[i,k] * du[ k ] * fract[k]
                    dWeights[index, 8] = 0 #% linear interpolation --> duu = 0
                    dWeights[index, 9] = ds[j,k] * dt[i,k] * (fract[k]**2) * du[k]


            # add weights
        for i in range(32):
            matrix_coefficient_Bspline_points[i] = pWeights[i]
            full_matrix_coefficient_points = full_matrix_coefficient_points + pWeights[i] * \
                                             self.local_matrix[int(control_points[i]),:]
            for k in range(10):
                matrix_coefficient_Bspline_der[i, k] = dWeights[i, k]
                full_matrix_coefficient_der[:, k] = \
                    full_matrix_coefficient_der[:, k] + dWeights[i, k] * \
                    self.local_matrix[int(control_points[i]),:]

        return full_matrix_coefficient_points, full_matrix_coefficient_der, control_points

    def calc_smoothing_matrix_DAffine(self, e_weights, e_groups=None):
        '''Changed by A.Mira to allow multiple elemts groups with different
        weights.
        e_groups - list of list with index of elements definind element group
        e_weight - nx3 array were n is the  list of groups,
        giving the weights for each group of elements'''

        # function that compiles the S'S matrix using D affine weights is a
        # 3 x1 vector containing the desired weight alog u, v and w
        # direction(element coordinates system)
        # Adaptati from calcDefSmoothingMatrix_RVLV3D.m
        if not self.build_mode:
            print('To compute the smoothing matrix the model should be '
                  'read with build_mode=True')
            return
        if e_groups == None:
            e_groups = [list(range(self.numElements))]

        e_weights = np.array(e_weights)
        if np.isscalar(e_groups[0]):
            e_groups = [e_groups]
        if len(e_weights.shape) ==1:
            e_weights = np.array([e_weights])
        # Creation of Gauss points

        xig, wg = generate_gauss_points(4)
        ngt = xig.shape[0]
        nft = 3
        nDeriv = [0, 1, 5]
        # d / dxi1, d / dxi2, d / dxi3 in position  1, 2 and 6
        dxi = 0.01
        # step in x space for finite difference calculation
        dxi1 = np.concatenate((np.ones((ngt,1)) * dxi,
                               np.zeros((ngt, 1)),
                               np.zeros((ngt, 1))),axis = 1)
        dxi2 = np.concatenate((np.zeros((ngt, 1)),
                               np.ones((ngt, 1)) * dxi,
                               np.zeros((ngt, 1))),axis = 1)
        dxi3 = np.concatenate((np.zeros((ngt, 1)),
                               np.zeros((ngt, 1)),
                               np.ones((ngt, 1)) * dxi),axis = 1)

        STSfull = np.zeros((self.numNodes, self.numNodes))
        Gx = np.zeros((self.numNodes, self.numNodes))
        Gy = np.zeros((self.numNodes, self.numNodes))
        Gz = np.zeros((self.numNodes, self.numNodes))

        dXdxi = np.zeros((3, 3))
        dXdxi11 = np.zeros((3, 3))
        dXdxi12 = np.zeros((3, 3))
        dXdxi21 = np.zeros((3, 3))
        dXdxi22 = np.zeros((3, 3))
        dXdxi31 = np.zeros((3, 3))
        dXdxi32 = np.zeros((3, 3))

        mBder = np.zeros((ngt, self.numNodes, 10))
        mBder11 = np.zeros((ngt, self.numNodes, 10))
        mBder12 = np.zeros((ngt, self.numNodes, 10))
        mBder21 = np.zeros((ngt, self.numNodes, 10))
        mBder22 = np.zeros((ngt, self.numNodes, 10))
        mBder31 = np.zeros((ngt, self.numNodes, 10))
        mBder32 = np.zeros((ngt, self.numNodes, 10))
        for et_index,et in enumerate(e_groups):
            weights = e_weights[et_index]
            for ne in et:
                print(ne)
                nr = 0
                Sk = np.zeros((3 * ngt*nft, self.numNodes, 3)) # storage for smoothing arrays

                # gauss points ' basis functions

                for j in range(ngt):
                    _, mBder[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                   xig[j, 2], ne , 0, 0,
                                                                   0)
                    _, mBder11[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                     xig[j, 2], ne ,
                                                                     dxi1[j, 0], dxi1[j, 1],
                                                                     dxi1[j, 2])
                    _, mBder12[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                     xig[j, 2], ne ,
                                                                     -dxi1[j, 0],
                                                                     -dxi1[j, 1],
                                                                     -dxi1[j, 2])
                    _, mBder21[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                     xig[j, 2], ne,
                                                                     dxi2[j, 0], dxi2[j, 1],
                                                                     dxi2[j, 2])
                    _, mBder22[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                     xig[j, 2], ne ,
                                                                     -dxi2[j, 0],
                                                                     -dxi2[j, 1],
                                                                     -dxi2[0, 2])
                    _, mBder31[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                     xig[j, 2], ne ,
                                                                     dxi3[j, 0], dxi3[j, 1],
                                                                     dxi3[j, 2])
                    _, mBder32[j,:,:], _ = self.evaluate_gauss_point(xig[j, 0], xig[j, 1],
                                                                     xig[j, 2], ne ,
                                                                     -dxi3[j, 0],
                                                                     -dxi3[j, 1],
                                                                     -dxi3[j, 2])


                # for all gauss pts ng
                for ng in range(ngt):
                    # calculate dX / dxi at Gauss pt and surrounding.
                    for nk,deriv in enumerate(nDeriv):
                        dXdxi[:, nk] = np.dot(mBder[ng,:, deriv],self.control_mesh)
                        dXdxi11[:, nk] = np.dot(mBder11[ng,:, deriv],self.control_mesh)
                        dXdxi12[:, nk] = np.dot(mBder12[ng,:, deriv],self.control_mesh)
                        dXdxi21[:, nk] = np.dot(mBder21[ng,:, deriv],self.control_mesh)
                        dXdxi22[:, nk] = np.dot(mBder22[ng,:, deriv],self.control_mesh)
                        dXdxi31[:, nk] = np.dot(mBder31[ng,:, deriv],self.control_mesh)
                        dXdxi32[:, nk] = np.dot(mBder32[ng,:, deriv],self.control_mesh)


                    g = np.linalg.inv(dXdxi)
                    g11 = np.linalg.inv(dXdxi11)
                    g12 = np.linalg.inv(dXdxi12)
                    g21 = np.linalg.inv(dXdxi21)
                    g22 = np.linalg.inv(dXdxi22)
                    g31 = np.linalg.inv(dXdxi31)
                    g32 = np.linalg.inv(dXdxi32)
                    h = np.zeros((3,3,3))
                    h[:,:, 0] = (g11 - g12) / (2 * dxi)
                    h[:,:, 1] = (g21 - g22) / (2 * dxi)
                    h[:,:, 2] = (g31 - g32) / (2 * dxi)

                    # 2 nd order derivatives[uu, uv, uw; uv, vv, vw; uw vw, ww]
                    pindex = np.array([[3, 2, 7],
                                       [2, 4, 6],
                                       [7, 6, 8]])

                    for nk in range(3): # derivatives
                        for nj in range(nft):
                            try:
                                Sk[nr,:, nk] = wg[ng] * ( \
                                            g[0, nj] * mBder[ng,:, pindex[nk, 0]]+
                                            g[1, nj] *mBder[ng,:, pindex[nk, 1]]+ \
                                            g[2, nj] * mBder[ng,:, pindex[nk, 2]] +
                                            h[0, nj, nk] * mBder[ng,:, 0] +
                                            h[1, nj,nk] * mBder[ng,:, 1]+
                                            h[2, nj, nk] * mBder[ng,:, 5])
                            except:
                                print('stop')
                            nr = nr + 1


                STS1    =   np.dot(Sk[:,:, 0].T, Sk[:,:,0])
                STS2    =   np.dot(Sk[:,:, 1].T, Sk[:,:,1])
                STS3    =   np.dot(Sk[:,:, 2].T, Sk[:,:,2])

                STS = (weights[0] * STS1) + (weights[1] * STS2) + (weights[2] * STS3)
                Gx = Gx + weights[0] * STS1
                Gy = Gy + weights[1] * STS2
                Gz = Gz + weights[2] * STS3

                # stiffness matrix
                STSfull = STSfull + STS


        GTSTSG = STSfull # I've already included G

        return GTSTSG, Gx, Gy, Gz

    def Get_centroid(self, isZero, I_EPI, J_EPI, K_EPI, x, y, z):
        ##This function is no longer necessary, as in a closed surface volume/mass calculations are robust to centroid selection and are arbitrary
        # calculates the centroid for the surface shape defined by I,J,K EPI (when built it was for epicardium hence epi)
        # isZero refers to whether the centroid should be dynamically calculated or just taken as point (0,0,0), these often produces different volumes
        # x, y and z are arrays of the respective xyz coordinates of the entire shape (incuding the target surface)
        # I, J and K EPI point to the vertices in xyz that correspond to the target surface. Every number in IJK_EPI points to an xyz vertex, the collection of three which makes a surface triangle
        # returns a list of three numbers, either (0,0,0) or the centroid point for the target surface (e.g. left/right ventricle)
        if isZero == True:
            return (0,0,0)
        else:
            centroid = [[], [], []]
            for mI, mJ, mK in zip(I_EPI, J_EPI, K_EPI):
                centroid[0].append(x[mI])
                centroid[0].append(x[mJ])
                centroid[0].append(x[mK])
                centroid[1].append(y[mI])
                centroid[1].append(y[mJ])
                centroid[1].append(y[mK])
                centroid[2].append(z[mI])
                centroid[2].append(z[mJ])
                centroid[2].append(z[mK])
            d = (np.median(centroid[0]), np.median(centroid[1]), np.median(centroid[2]))
            return d

    def Get_tetrahedron_vol_CM(self, a, b, c, d):
        #Calculates volume of tetrahedron abcd, where abc is the three surface point vertices and d is the fixed centroid for the shape
        #Utilised in Mass/volume calculations where it is returned in ml3 once divided by 1000
        bxdx = b[0] - d[0]
        bydy = b[1] - d[1]
        bzdz = b[2] - d[2]
        cxdx = c[0] - d[0]
        cydy = c[1] - d[1]
        czdz = c[2] - d[2]

        vol = ((a[2] - d[2]) * ((bxdx*cydy) - (bydy*cxdx))) +\
              ((a[1] - d[1]) * ((bzdz*cxdx) - (bxdx*czdz))) +\
              ((a[0] - d[0]) * ((bydy*czdz) - (bzdz*cydy)))
        vol = vol/6
        return vol

    def Get_ventricular_vol(self, centroid_isZero):
        #Calculates the ventricular volumes of the left and right ventricles in ml3, returning as a tuple (LVvol,RVvol).
        #Centroid_isZero is redundant and used for checking if surface is closed, now will not affect output
        x = np.array(self.et_pos[:, 0]).T
        y = np.array(self.et_pos[:, 1]).T
        z = np.array(self.et_pos[:, 2]).T
        LvSurfaces = [Surface.LV_ENDOCARDIAL, Surface.MITRAL_VALVE, Surface.AORTA_VALVE] #surfaces that comprise LV closed surface
        RvSurfaces = [Surface.RV_FREEWALL, Surface.TRICUSPID_VALVE, Surface.PULMONARY_VALVE] #surfaces that comprise RV closed surface
        if centroid_isZero == True: #redundant, as discussed. Used for debugging closed surface.
            D = [0,0,0]
        else:
            D = self.Get_centroid(isZero=False, I_EPI=np.asarray(self.et_indices[self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[0]:
                                                                             self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1], 0]),
                              J_EPI=np.asarray(self.et_indices[self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[0]:
                                                                             self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1], 1]),
                              K_EPI=np.asarray(self.et_indices[self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[0]:
                                                                             self.get_surface_vertex_start_end_index(Surface.LV_ENDOCARDIAL)[1], 2]),
                                x=x, y=y, z=z)
        LVvol = 0
        for i in LvSurfaces:
            seStart = self.get_surface_start_end_index(surface_name=i)[0] #index where surface i start
            seEnd = self.get_surface_start_end_index(surface_name=i)[1] #index where surface i ends
            for se in range(seStart,seEnd+1): #range of surface triangles that comprise surface i
                indices = self.et_indices[se] #each se in range start-end corresponds to a set of 3 indices in et_pos
                Pts = self.et_pos[indices] #each of these 3 indices corresponds to a point, defined by x,y,z co-ords. Pts is 3x3 matrix of those points, and forms surface triangle.
                LVvol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D) #these three triangle points and centroid D form tetrahedron, volume of which is calculated
        LVvol /= 1000 #by iterating through all surface triangles that comprise LV and adding the volumes, you calculate the volume of the closed ventricle
        RVvol = 0
        for i in RvSurfaces: #same concept as for LV, different surfaces
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            for se in range(seStart,seEnd+1):
                indices = self.et_indices[se]
                Pts = self.et_pos[indices]

                RVvol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        RVvol /= 1000

        RVsVol = 0 #RV septum has inverted normal; volume of RVS must be subtracted from RV_vol, rather than added like other surfaces
        seStart = self.get_surface_start_end_index(surface_name=Surface.RV_SEPTUM)[0]
        seEnd = self.get_surface_start_end_index(surface_name=Surface.RV_SEPTUM)[1]
        for se in range(seStart, seEnd + 1):
            indices = self.et_indices[se]
            Pts = self.et_pos[indices]
            RVsVol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        RVsVol /= 1000

        return(LVvol, RVvol-RVsVol) #RVS subtracted from RV to give proper RV_vol, LV has no inverted normals. Both return in ml3

    def Get_myocardial_mass(self, LVvol, RVvol):
        #Calculates volume of closed epicardial surface, subtracts ventricular volumes and multiplies myocardium density to return in grams
        D = [0,0,0] #arbitrary centroid point
        LVMyoVol = 0
        RVMyoVol = 0
        #LV Epicardium
        for se in range(2416,len(self.et_indices_EpiLVRV)): #same style as volume calcs, get volume of LV epicardium defined by EpiLVRV
            indices = self.et_indices_EpiLVRV[se]
            Pts = self.et_pos[indices]
            LVMyoVol += self.Get_tetrahedron_vol_CM(Pts[0],Pts[1],Pts[2],D)
        Lv_MyoVol_sum = LVMyoVol/1000

        #RV Epicardium
        for se in range(0,2416):
            indices = self.et_indices_EpiLVRV[se]
            Pts = self.et_pos[indices]
            RVMyoVol += self.Get_tetrahedron_vol_CM(Pts[0],Pts[1],Pts[2],D)
        RvMyoVol_sum = RVMyoVol/1000

        #ThruWall surface
        VolThru = 0 #thruwall surface divides LV and RV through septum (which doesn't cover epicardium to close surfaces separately)
        for se in range(0,len(self.et_indices_thruWall)):
            indices = self.et_indices_thruWall[se]
            Pts = self.et_pos[indices]
            VolThru += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        Lv_MyoVol_sum -= VolThru/1000 #ThruWall normals inverted for LV, normal for RV. Therefore subtract vol from LV, add to RV
        RvMyoVol_sum += VolThru/1000

        #Mitral/aortic valves
        vol = 0 #valve points for LV
        for i in [Surface.MITRAL_VALVE,Surface.AORTA_VALVE]:
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            for se in range(seStart, seEnd + 1):
                indices = self.et_indices[se]
                Pts = self.et_pos[indices]
                vol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        Lv_MyoVol_sum += vol/1000

        #Pulmonary/tricuspid valves
        vol = 0 #valve points for RV
        for i in [Surface.PULMONARY_VALVE,Surface.TRICUSPID_VALVE]:
            seStart = self.get_surface_start_end_index(surface_name=i)[0]
            seEnd = self.get_surface_start_end_index(surface_name=i)[1]
            for se in range(seStart, seEnd + 1):
                indices = self.et_indices[se]
                Pts = self.et_pos[indices]
                vol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        RvMyoVol_sum += vol/1000

        #Septum
        vol = 0
        seStart= self.get_surface_start_end_index(Surface.RV_SEPTUM)[0]
        seEnd = self.get_surface_start_end_index(Surface.RV_SEPTUM)[1]
        for se in range(seStart, seEnd + 1):
            indices = self.et_indices[se]
            Pts = self.et_pos[indices]
            vol += self.Get_tetrahedron_vol_CM(Pts[0], Pts[1], Pts[2], D)
        Lv_MyoVol_sum += vol/1000 #Septum normals face LV, inverted for RV, therefore vol added to LV and subtracted from RV
        RvMyoVol_sum -= vol/1000

        #Mass calculations
        LVmass = (Lv_MyoVol_sum - LVvol) * 1.05 #1.05 is density of myocardial mass
        RVmass = (RvMyoVol_sum - RVvol) * 1.05 #calculation is volume of each ventricle epicardium-ventricular volume, * myocardium density
        return (LVmass, RVmass) #returns tuple of LV_myo_mass, RV_myo_mass, both in grams

    def update_control_mesh(self,new_control_mesh):
        self.control_mesh = new_control_mesh
        self.et_pos = np.linalg.multi_dot(
            [self.matrix, self.control_mesh])


    @staticmethod
    def subdivide_mesh(n, points, cells):
        mesh = trimesh.Trimesh(points, cells)
        for i in range(n):
            mesh = mesh.subdivide()
        mesh = io.Mesh(mesh.vertices, {'triangle': mesh.faces})
        return mesh


    def get_bv_surface_mesh(self, subdivisions=0):
        points = self.et_pos
        cells = self.et_indices
        surfs = self.surfs

        # Surfaces defining BiV
        bv_surfs = [0,1,2,3,8,9]
        bv_marker = np.isin(surfs, bv_surfs)

        if subdivisions > 0:
            mesh = self.subdivide_mesh(subdivisions, points, cells[bv_marker])
        else:
            mesh = io.Mesh(points, {'triangle': cells[bv_marker]})

        # Extracting valves only
        valve_elems = self.valve_elems
        valve_cells = cells[valve_elems]

        if subdivisions > 0:
            valve_mesh = self.subdivide_mesh(subdivisions, points, valve_cells)
        else:
            valve_mesh = io.Mesh(points, {'triangle': valve_cells})

        # Extracting septum
        sep_marker = surfs == 1
    
        if subdivisions > 0:
            septum_mesh = self.subdivide_mesh(subdivisions, points, cells[sep_marker])
        else:
            septum_mesh = io.Mesh(points, {'triangle': cells[sep_marker]})

        return mesh, valve_mesh, septum_mesh




    def get_lv_rv_surface_mesh(self):
        points = self.et_pos
        cells = self.et_indices
        surfs = self.surfs

        # Surfaces defining BiV
        lv_surfs = [0,1,8,9]
        rv_surfs = [2,3,9]

        lv_marker = np.isin(surfs, lv_surfs)
        lv_mesh = io.Mesh(points, {'triangle': cells[lv_marker]})

        rv_marker = np.isin(surfs, rv_surfs)
        rv_mesh = io.Mesh(points, {'triangle': cells[rv_marker]})

        return lv_mesh, rv_mesh


    def get_long_axis_landmarks(self):
        mv = self.et_pos[self.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1]]
        apex = self.et_pos[self.apex_epi_index]
        la_landmarks = np.vstack([apex, mv])
        return la_landmarks
