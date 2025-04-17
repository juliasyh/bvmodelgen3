import numpy as np
import pandas as pd
import warnings
import os
import meshio as io

#local imports
from . import fitting_tools as tools
from .surface_enum import  ContourType

SAMPLED_CONTOUR_TYPES = [ContourType.LAX_LV_ENDOCARDIAL,
                         ContourType.LAX_LV_EPICARDIAL,
                         ContourType.SAX_LV_ENDOCARDIAL,
                         ContourType.SAX_LV_EPICARDIAL,
                         ContourType.LAX_LA,
                         ContourType.SAX_RV_ENDOCARDIAL,
                         ContourType.LAX_RV_ENDOCARDIAL,
                         ContourType.SAX_RV_EPICARDIAL,
                         ContourType.SAX_RV_EPICARDIAL,
                         ContourType.SAX_RV_FREEWALL,
                         ContourType.LAX_RV_FREEWALL,
                         ContourType.SAX_RV_SEPTUM,
                         ContourType.LAX_RV_SEPTUM,
                         ContourType.SAX_RV_OUTLET]
UNSAMPLED_CONTOUR_TYPES = [ContourType.MITRAL_VALVE,
                           ContourType.TRICUSPID_VALVE,
                           ContourType.AORTA_VALVE,
                           ContourType.PULMONARY_VALVE,
                            ContourType.APEX_ENDO_POINT,
                             ContourType.APEX_EPI_POINT,
                            ContourType.RV_INSERT]



##Author : Charlène Mauger, University of Auckland, c.mauger@auckland.ac.nz
class GPDataSet(object):
    """ This class reads a dataset. A DataSet object has the following properties:

    Attributes:
        case: case name
        mitral_centroid: centroid of the 3D contour points labelled as mitral valve
        tricuspid_centroid: centroid of the 3D contour points labelled as tricuspid valve
        aorta_centroid: centroid of the 3D contour points labelled as aortic valve
        pulmonary_centroid: centroid of the 3D contour points labelled as pulmonic valve
        number_of_slice: number of 2D slices
        number_of_time_frame: number of time frames
        points_coordinates: 3D coordinates of the contour points
    """

    def __init__(self, contour_filename, case ='default' , sampling = 1,
                 time_frame_number =None ):
        """ Return a DataSet object. Each point of this dataset is characterized by
        its 3D coordinates ([Evenly_spaced_points[:,0],Evenly_spaced_points[:,1],
        Evenly_spaced_points[:,2]]), the slice it belongs to (slice_number) and the surface
        its belongs to (ContourType)

            Input:
                filename: filename is the file containing the 3D contour points coordinates,
                            labels and time frame (see example GPfile.txt).
                filenameInfo: filename is the file containing dicom info
                            (see example SliceInfoFile.txt).
                case: case number
                time_frame_number: time frame #
        """
        # arrays

        self.points_coordinates = np.empty((0, 3))
        self.contour_type = np.empty((1))
        self.slice_number = np.empty((1))
        self.weights = np.empty((1))
        self.time_frame =None
        self.number_of_slice = 0
        self.frames = {}

        # strings
        self.case = case

        # scalars
        self.time_frame = time_frame_number
        self._read_contour_file(contour_filename,time_frame_number,sampling)
        self._initialize_landmarkds()
        self.ContourType = ContourType
        

    def _read_contour_file(self, filename, time_frame_number =None, sampling = 1):
        '''add  by A. Mira 02/2020'''
        # column num 3 of my datset is a space
        if not os.path.exists(filename):
            warnings.warn('Contour files does not exist')
            return
        P = []
        slices =[]
        contypes = []
        weights = []
        time_frame =[]
        try:
            data = pd.read_csv(open(filename), sep='\t', header=None)
            for line_index,line in enumerate(data.values[1:]):
                P.append([float(x) for x in line[:3]])
                slices.append(int(line[4]))
                contypes.append(line[3])
                weights.append(float(line[5]))
                try:
                    time_frame.append(int(float(line[6])))
                except:
                    time_frame.append(time_frame_number)  # data[6]

            P = np.array(P)

            slices = np.array(slices)
            contypes = np.array(contypes)
            weights = np.array(weights)
            time_frame = np.array(time_frame)

        except ValueError:
            print("Wrong file format: {0}".format(filename))



        if time_frame_number is not None:
            valid_contour_index = np.array(time_frame == time_frame_number)
            if (np.sum(valid_contour_index) == 0):
                warnings.warn('Wrong time frame number')
                return

            P = P[valid_contour_index, :]
            slices = slices[valid_contour_index]
            contypes = contypes[valid_contour_index]
            weights = weights[valid_contour_index]

        contypes = self._convert_contour_types(contypes)
        # increment contours points which don't need sampling

        valid_contour_index = np.array([x in UNSAMPLED_CONTOUR_TYPES
                                        for x in contypes])

        self.points_coordinates = P[valid_contour_index]
        self.contour_type = contypes[valid_contour_index]
        self.slice_number = slices[valid_contour_index]
        self.weights = weights[valid_contour_index]
        del_index =  list(np.where(valid_contour_index)[0])
        P = np.delete(P,del_index, axis = 0)
        contypes = np.delete(contypes, del_index)
        slices = np.delete(slices, del_index)
        weights = np.delete(weights, del_index)

        self.number_of_slice = len(self.slice_number)  # slice index starting with 0

        self._sample_contours(P, slices, contypes ,weights,sampling) # there are
        # too many
        # points extracted from cvi files.  To reduce computation time,
        # the contours points are sampled

        self.number_of_slice = max(self.slice_number) + 1



    def _sample_contours(self, points, slices, contypes, weights, sample):

        for j in np.unique(slices):  # For slice i, extract evenly
            # spaced point for all type
            for contour_index, contour_type in enumerate(
                    SAMPLED_CONTOUR_TYPES):

                C = points[(contypes == contour_type) & (slices == j), :]
                C_weights = weights[(contypes == contour_type) & (slices == j)]

                if len(C) > 0:
                    # sort the points by euclidean distance from the
                    # previous point

                    Cx_index,Cx = tools.sort_consecutive_points(C)
                    if len(Cx.shape) == 1:
                        Cx = Cx.reshape(1,-1)


                    self.points_coordinates = np.vstack(
                        (self.points_coordinates, Cx[0::sample, :]))
                    self.slice_number = np.hstack(
                        (self.slice_number, [j] * len(Cx[0::sample, :])))
                    self.contour_type = np.hstack((self.contour_type,
                                                   [contour_type] * len(
                                                       Cx[0::sample, :])))
                    self.weights = np.hstack(
                        (self.weights, C_weights[Cx_index[0::sample]]))

    def _initialize_landmarkds(self):
        " add by A.Mira on 01/2020"
        # calc valve centroids

        P = self.points_coordinates
        mitral_index = (self.contour_type == ContourType.MITRAL_VALVE)
        if (np.sum(mitral_index)> 0):
            self.mitral_centroid = P[mitral_index, :].mean(axis=0)

        tricuspid_index = (self.contour_type == ContourType.TRICUSPID_VALVE)
        if np.sum(tricuspid_index)>0:
            self.tricuspid_centroid = P[tricuspid_index, :].mean(axis=0)

        aorta_contour_index = (self.contour_type == ContourType.AORTA_VALVE)
        if np.sum(aorta_contour_index) >0 :
            self.aorta_centroid = P[aorta_contour_index,:].mean(axis=0)

        pulmonary_index = (self.contour_type == ContourType.PULMONARY_VALVE)
        if np.sum(pulmonary_index) >0:
            self.pulmonary_centroid = P[pulmonary_index, :].mean(axis=0)

        apex_endo_index = (self.contour_type == ContourType.APEX_ENDO_POINT)
        if np.sum(apex_endo_index)> 0:
            self.apex_endo = P[apex_endo_index, :]

            if len(self.apex_endo) > 0:
                self.apex_endo = self.apex_endo[0, :]
        apex_epi_index = (self.contour_type == ContourType.APEX_EPI_POINT)
        if np.sum(apex_endo_index)> 0:
            self.apex_epi = P[apex_epi_index, :]

            if len(self.apex_epi) > 0:
                self.apex_epi = self.apex_epi[0, :]

    @staticmethod
    def _convert_contour_types(contours):
        " add by A.Mira on 01/2020"
        # convert contours from string type to Contour enumeration
        # type

        #print(contours)
        new_contours =  np.empty(contours.shape[0], dtype=ContourType)
        for contour_type in ContourType:
            new_contour_index = np.where(contours == \
                                contour_type.value)[0]
            new_contours[new_contour_index] = contour_type
        return new_contours


    def create_rv_epicardium(self, rv_thickness):
        """ This function generates phantom points for the RV epicardium.
        Epicardium of the RV free wall was not manually contoured in our dataset,
         but it is better to have them when customizing the surface mesh.
        RV epicardial phantom points are estimated by extending the RV endocardium
        contour points by a fixed distance (3mm from the literature).
        If your dataset contains RV epicardial point, you can comment this function
        Input:
            rv_thickness : thickness of the wall to be created
        Output:
            rv_epi: RV epicardial phantom points
        """

        # RV_wall_thickness: normal value from literature
        rv_epi = []
        rv_epi_slice =[]
        rv_epi_contour =[]
        valid_contours = [ContourType.SAX_RV_FREEWALL,
                          ContourType.SAX_RV_OUTLET,
                          ContourType.LAX_RV_FREEWALL]
        epi_contours = [ContourType.SAX_RV_EPICARDIAL,
                        ContourType.LAX_RV_EPICARDIAL]

        for i in np.unique(self.slice_number):

            # For each slice, find centroid cloud point RV_FREEWALL
            # Get contour points

            valid_index = ([x in valid_contours[:2] for x in
                             self.contour_type]) * (self.slice_number == i)
            points_slice = self.points_coordinates[valid_index  , :]

            if len(points_slice) > 0:
                slice_centroid = points_slice.mean(axis=0)
                contour_index =0
            else:
                points_slice = self.points_coordinates[(self.contour_type ==
                                valid_contours[2])
                                    & (self.slice_number == i),:]
                if len(points_slice) > 0:
                    slice_centroid = points_slice.mean(axis=0)
                    contour_index =1
                else:
                    continue

            for j in points_slice:
                # get direction
                direction = j[0:3] - slice_centroid
                direction = direction / np.linalg.norm(direction)
                # Move j along direction by rv_thickness
                new_position = np.add(j[0:3],
                                      np.array([rv_thickness * direction[0],
                                                rv_thickness * direction[1],
                                                rv_thickness * direction[2]
                                                ]))
                rv_epi.append(np.asarray([new_position[0], new_position[1],
                                          new_position[2]]))
                rv_epi_slice.append(i)
                rv_epi_contour.append(epi_contours[contour_index])

        self.add_data_points(np.asarray(rv_epi), np.array(rv_epi_contour),
                             np.array(
                                 rv_epi_slice),
                                 [1] * len(rv_epi))

        return np.asarray(rv_epi),np.array(rv_epi_contour),np.array(
            rv_epi_slice)


    def add_data_points(self, points, contour_type, slice_number, weights ):
        """
        add new contour points to a data set
        input:
            points: nx3 array with points coordinates
            contour_type: n list with the contour type for each point
            slice_number: n list with the slice number for each point
        """
        if len(points) == len(contour_type) == len(slice_number) == len(weights):
            self.points_coordinates = np.vstack((self.points_coordinates, points))
            self.slice_number = np.hstack((self.slice_number, slice_number))
            self.contour_type = np.hstack((self.contour_type,contour_type))
            self.weights = np.hstack((self.weights, weights))
        else:
            print('In add_data_point input vectors should have the same lenght')

    def identify_mitral_valve_points(self):
        """ This function matches each Mitral valve point with the LAX slice it
        was  extracted
        from.
            Input:
                None
            Output:
                None. slice_number for each Mitral valve point is changed to
                the corresponding LAX slice number
        """

        mitral_points = self.points_coordinates[(self.contour_type ==
                                                 ContourType.MITRAL_VALVE), :]

        new_mitral_points = np.zeros((len(mitral_points), 3))
        corresponding_slice = []
        it = 0
        for slice_id in np.unique(self.slice_number):

            LAX = self.points_coordinates[
                  (self.contour_type == ContourType.LAX_LA)
                  & (self.slice_number == slice_id), :]
            if len(LAX) > 0:
                Corr = np.zeros((len(mitral_points), 1))
                NP = np.zeros((len(mitral_points), 3))
                Sl = np.zeros((len(mitral_points), 1))
                # Find corresponding BP on this slices - should be two

                for points in range(len(mitral_points)):
                    i = (np.square(mitral_points[points, :] - LAX)).sum(1).argmin()
                    Corr[points] = np.linalg.norm(
                        LAX[i, :] - mitral_points[points, :])
                    NP[points] = LAX[i, :]
                    Sl[points] = slice_id

                index = Corr.argmin()
                new_mitral_points[it, :] = NP[index, :]
                corresponding_slice.append(float(Sl[index]))

                NP = np.delete(NP, index, 0)
                Sl = np.delete(Sl, index, 0)
                Corr = np.delete(Corr, index, 0)
                it = it + 1

                index = Corr.argmin()
                new_mitral_points[it, :] = NP[index, :]
                corresponding_slice.append(float(Sl[index]))
                it = it + 1

        indexes = np.where((self.contour_type == ContourType.MITRAL_VALVE))
        self.points_coordinates[indexes] =  new_mitral_points
        self.contour_type[indexes] = [ContourType.MITRAL_VALVE] * len(new_mitral_points)
        self.slice_number[indexes] = corresponding_slice

    def create_valve_phantom_points(self, n, contour_type):
        """ This function creates mitral phantom points by fitting a circle to the mitral points
        from the DataSet
            Input:
                n: number of phantom points we want to create
            Output:
                P_fitcircle: phantom points
        """
        new_points=[]
        valid_contour_types = np.array([ContourType.TRICUSPID_VALVE,
                                        ContourType.MITRAL_VALVE,
                                        ContourType.PULMONARY_VALVE,
                                        ContourType.AORTA_VALVE])
        if not (contour_type in valid_contour_types):
            return []

        if contour_type == ContourType.MITRAL_VALVE:
            case = 'mv'
        elif contour_type == ContourType.TRICUSPID_VALVE:
            case = 'tv'
        elif contour_type == ContourType.AORTA_VALVE:
            case = 'av'

        valve_points = self.points_coordinates[self.contour_type ==
                                            contour_type, :]
        if case == 'mv':
            av_valve_points = self.points_coordinates[self.contour_type ==
                                                     ContourType.AORTA_VALVE, :]
            if len(av_valve_points) == 0:
                mv_av_correction = False
            else:
                mv_av_correction = True

        if len(valve_points) > n:# if we have enough points to define the
        # contour,
            # better keep the contour itself
            return valve_points
        if len(valve_points) == 0:
            return  new_points

        # Coordinates of the 3D points
        P = np.array(valve_points)

        distance = [np.linalg.norm(P[i]-P[j]) for i in range(len(P)) for j
                    in range(len(P))]

        if len(distance) == 0:
            return np.empty((0,3))

        valid_points = False
        if max(distance) > 10 and len(valve_points)< 3:
            valid_points = True


        if valid_points:
            if contour_type == ContourType.AORTA_VALVE:
                vector = valve_points[1] - valve_points[0]
                vector2 = self.aorta_centroid - self.mitral_centroid

                aux = np.cross(vector, vector2)
                normal = np.cross(vector, aux)
            else:
                # Define a vector from one point to the other
                vector = valve_points[1] - valve_points[0]
                vector = vector/np.linalg.norm(vector)
                
                # cross it with the LA vector
                la_vector = self.mitral_centroid - self.apex_endo
                la_vector = la_vector/np.linalg.norm(la_vector)
                
                aux = np.cross(la_vector, vector)

                normal = np.cross(vector, aux)
            
            center = np.mean(valve_points, axis=0)
            points = valve_points - center
            angles = np.linspace(0, 2*np.pi, n)
            new_points = np.zeros([len(angles), 3])
            cont = 0
            for i in range(len(angles)):
                # Avoid adding same points
                if angles[i] == 0: continue
                if angles[i] == np.pi: continue
                rot_vec = tools.rodrigues_rot_angle(points[0], normal, angles[i])
                new_points[cont] = center + rot_vec
                cont += 1
            new_points = new_points[0:cont]


        else:
            P_mean = P.mean(axis=0)
            P_centered = P - P_mean
            U, s, V = np.linalg.svd(P_centered)

            # Normal vector of fitting plane is given by 3rd column in V
            # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
            normal_valve = V[2, :]

            # -------------------------------------------------------------------------------
            # (2) Project points to coords X-Y in 2D plane
            # -------------------------------------------------------------------------------
            P_xy = tools.rodrigues_rot(P_centered, normal_valve, [0, 0, 1])
            center, axis_l, rotation = tools.fit_elipse_2d(P_xy[:,:2])

            # Check aspect ratio of the ellipse
            axis_l = np.sort(axis_l)
            if axis_l[1]/axis_l[0] > 2:
                axis_l[0] = axis_l[1]/1.5
                t = np.array([0, np.pi])
                new_points = tools.generate_2Delipse_by_vectors(t, center, axis_l,
                                                        rotation)
                P_xy = np.vstack((P_xy[:,:2], new_points))
                center, axis_l, rotation = tools.fit_elipse_2d(P_xy)
            
            # --- Generate points for fitting circle
            t = np.linspace(-np.pi, np.pi, n)
        
            new_points = tools.generate_2Delipse_by_vectors(t, center, axis_l,
                                                    rotation)
            new_points = np.array([new_points[:, 0], new_points[:, 1],
                                [0] * new_points.shape[0]]).T
            new_points = tools.rodrigues_rot(new_points,[0,0,1],
                                            normal_valve)+ P_mean
            
            
        # If MV, we need to warp the ellipsoid to match the bridge better
        if case == 'mv' and mv_av_correction:
            # Find what MV point is closer to AV
            from scipy.spatial.distance import cdist
            dist = cdist(av_valve_points, valve_points)
            min_index = np.unravel_index(np.argmin(dist), dist.shape)
            av_min_index = min_index[0]
            mv_min_index = min_index[1]

            vector = valve_points[mv_min_index] - av_valve_points[av_min_index] 
            vector = vector/np.linalg.norm(vector)


            # Grab coordinates in plane of the new points
            vec1 = vector
            vec3 = normal_valve
            vec2 = np.cross(vec3, vec1)
            vec2 = vec2/np.linalg.norm(vec2)
            vec3 = np.cross(vec1, vec2)
            Q = np.array([vec1, vec2, vec3])

            # Project points to plane made by vec1 and vec2
            xyz = (new_points - P_mean)@Q.T
            xy = xyz[:, 0:2]


            mv_bridge_point = valve_points[mv_min_index]
            mv_bridge_zcoord = np.dot(mv_bridge_point - P_mean, vec3)
            mv_bridge_xcoord = np.dot(mv_bridge_point - P_mean, vec1)
            mv_bridge_ycoord = np.dot(mv_bridge_point - P_mean, vec2)
            mv_bridge_xy = np.array([mv_bridge_xcoord, mv_bridge_ycoord]).T


            dist_to_bridge = xy[:,0] - mv_bridge_xy[0]
            dist_to_bridge = (dist_to_bridge - dist_to_bridge.min()) / (
                    dist_to_bridge.max() - dist_to_bridge.min())
            dist_to_bridge = 1 - dist_to_bridge
            
            weigth_func = 1/(1 + np.exp(-(dist_to_bridge-0.85)*7))
            new_zcoord = xyz[:,2] * (1-weigth_func) + mv_bridge_zcoord * weigth_func
            
            weigth_func = 1/(1 + np.exp(-(dist_to_bridge-0.8)*10))
            new_xcoord = xyz[:,0] * (1-weigth_func) + mv_bridge_xcoord * weigth_func

            
            # plt.figure()
            # plt.scatter(xy[:, 0], xy[:, 1], c=new_zcoord)
            # plt.scatter(mv_bridge_point[0], mv_bridge_point[1])

            # Go back to 3D
            new_points = np.array([new_xcoord, xyz[:,1], new_zcoord]).T
            new_points = new_points@Q + P_mean

            # from mpl_toolkits.mplot3d import Axes3D
            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c='b', marker='o')
            # ax.scatter(new_points2[:, 0], new_points2[:, 1], new_points2[:, 2], c='r', marker='o')
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()

        # insert new points in the dataset
        # output type depnes on the input contour type
        # and weight are computed as mean weight of the valve points
        if len(new_points) >0:
            if contour_type == ContourType.MITRAL_VALVE:
                output_type = ContourType.MITRAL_PHANTOM
            elif contour_type == ContourType.TRICUSPID_VALVE:
                output_type = ContourType.TRICUSPID_PHANTOM
            elif contour_type == ContourType.AORTA_VALVE:
                output_type = ContourType.AORTA_PHANTOM
            elif contour_type == ContourType.PULMONARY_VALVE:
                output_type = ContourType.PULMONARY_PHANTOM


            weight_MV = self.weights[self.contour_type == contour_type].mean()

            self.add_data_points(new_points,
                                [output_type] * len(new_points),
                                [-1] * len(new_points),
                                [weight_MV] * len(new_points))

        return new_points
    

    def assign_weights(self, new_weights):
        self.weights[
            self.contour_type == ContourType.APEX_ENDO_POINT] \
            *= new_weights['apex_endo']
        self.weights[
            self.contour_type == ContourType.APEX_EPI_POINT] \
            *= new_weights['apex_epi']
        self.weights[
            self.contour_type == ContourType.MITRAL_VALVE] \
            *= new_weights['mv']
        self.weights[
            self.contour_type == ContourType.MITRAL_PHANTOM] \
            *= new_weights['mv_phantom']
        self.weights[
            self.contour_type == ContourType.TRICUSPID_VALVE] \
            *= new_weights['tv']
        self.weights[
            self.contour_type == ContourType.TRICUSPID_PHANTOM] \
            *= new_weights['tv_phantom']
        self.weights[
            self.contour_type == ContourType.AORTA_VALVE] \
            *= new_weights['av']
        self.weights[
            self.contour_type == ContourType.AORTA_PHANTOM] \
            *= new_weights['av_phantom']
        self.weights[
            self.contour_type == ContourType.PULMONARY_VALVE] \
            *= new_weights['pv']
        self.weights[
            self.contour_type == ContourType.PULMONARY_PHANTOM] \
            *= new_weights['pv_phantom']
        self.weights[
            self.contour_type == ContourType.RV_INSERT] \
            *= new_weights['rv_insert']
        self.weights[
            self.contour_type == ContourType.LAX_RV_ENDOCARDIAL] \
            *= new_weights['la_rv_endo']
        self.weights[
            self.contour_type == ContourType.LAX_RV_EPICARDIAL] \
            *= new_weights['la_rv_epi']
        self.weights[
            self.contour_type == ContourType.LAX_LV_ENDOCARDIAL] \
            *= new_weights['la_lv_endo']
        self.weights[
            self.contour_type == ContourType.LAX_LV_EPICARDIAL] \
            *= new_weights['la_lv_epi']
        self.weights[
            self.contour_type == ContourType.SAX_LV_EPICARDIAL] \
            *= new_weights['sa_lv_epi']
        self.weights[
            self.contour_type == ContourType.SAX_LV_ENDOCARDIAL] \
            *= new_weights['sa_lv_endo']


    def PlotDataSet(self, contours_to_plot =[]):
        """ This function plots this entire dataset.
            Input:
                Con
            Output:
                traces for figure
        """
        # contours lines types
        contour_lines = np.array([ContourType.TRICUSPID_PHANTOM,
                                  ContourType.LAX_RA,
                                  ContourType.SAX_RV_FREEWALL,
                                  ContourType.LAX_RV_FREEWALL,
                                  ContourType.SAX_RV_SEPTUM,
                                  ContourType.LAX_RV_SEPTUM,
                                  ContourType.SAX_RV_EPICARDIAL,
                                  ContourType.LAX_RV_EPICARDIAL,
                                  ContourType.LAX_RV_ENDOCARDIAL,
                                  ContourType.SAX_RV_ENDOCARDIAL,
                                  ContourType.SAX_RV_OUTLET,
                                  ContourType.PULMONARY_PHANTOM,
                                  ContourType.AORTA_PHANTOM,
                                  ContourType.MITRAL_PHANTOM,
                                  ContourType.LAX_LA,
                                  ContourType.SAX_LV_EPICARDIAL,
                                  ContourType.LAX_LV_EPICARDIAL,
                                  ContourType.SAX_LV_ENDOCARDIAL,
                                  ContourType.LAX_LV_ENDOCARDIAL,
                                  ])
        lines_color_map = np.array(["rgb(128,0,128)","rgb(186,85,211)",
                                    "rgb(0,0,205)", "rgb(65,105,225)",
                                    "rgb(139,0,139)",  "rgb(153,50,204)",
                                    "rgb(0,191,255)", "rgb(30,144,255)",
                                    "rgb(0,0,205)", "rgb(65,105,225)",
                                    "rgb(0,206,209)","rgb(95,158,160)",
                                    "rgb(128,0,0)", "rgb(205,92,92)",
                                    "rgb(220,20,60)","rgb(255,127,80)",
                                    "rgb(85,107,47)","rgb(50,205,50)",
                                    "rgb(85,107,47)","rgb(50,205,50)"])
        # points types
        contour_points = np.array([ContourType.RV_INSERT, ContourType.APEX_ENDO_POINT,
                                   ContourType.APEX_EPI_POINT,
                                   ContourType.MITRAL_VALVE,
                                   ContourType.TRICUSPID_VALVE,
                                   ContourType.AORTA_VALVE,
                                   ContourType.PULMONARY_VALVE])
        points_color_map = np.array(["rgb(255,20,147)", "rgb(0,191,255)", "rgb(0,191,255)",
                                     "rgb(255,0,0)","rgb(128,0,128)",
                                     "rgb(0,255,0)", "rgb(0,43,0)",])

        if not isinstance(contours_to_plot, list):
            contours_to_plot = [contours_to_plot]
        if len(contours_to_plot) == 0:
            contours_to_plot = contour_lines + contour_points

        contourPlots =[]
        for  contour in contours_to_plot:
            contour_index = np.where(contour_lines == contour)[0]
            points_size =2

            if len(contour_index)== 0:
                contour_index = np.where(contour_points == contour)[0]
                points_size = 5
                if len(contour_index)==1:
                    points_color= points_color_map[contour_index][0]
            else:
                points_color = lines_color_map[contour_index][0]

            if len(contour_index)>0:

                contourPlots = contourPlots + tools.Plot3DPoint(
                    self.points_coordinates[
                                np.where(np.asarray(
                                    self.contour_type) == contour)],
                            points_color, points_size, contour.value)
        return contourPlots



    def to_vertex_mesh(self):
        ctype = np.zeros(len(self.points_coordinates))
        type2int = {ContourType.LAX_LV_ENDOCARDIAL:1,
                    ContourType.LAX_LV_EPICARDIAL:2,
                    ContourType.SAX_LV_ENDOCARDIAL:3,
                    ContourType.SAX_LV_EPICARDIAL:4,
                    ContourType.LAX_LA:5,
                    ContourType.SAX_RV_ENDOCARDIAL:6,
                    ContourType.LAX_RV_ENDOCARDIAL:7,
                    ContourType.SAX_RV_EPICARDIAL:8,
                    ContourType.SAX_RV_EPICARDIAL:9,
                    ContourType.SAX_RV_FREEWALL:10,
                    ContourType.LAX_RV_FREEWALL:11,
                    ContourType.SAX_RV_SEPTUM:12,
                    ContourType.LAX_RV_SEPTUM:13,
                                 ContourType.SAX_RV_OUTLET:14,
                                 ContourType.MITRAL_VALVE:15,
                                   ContourType.TRICUSPID_VALVE:16,
                                   ContourType.AORTA_VALVE:17,
                                   ContourType.PULMONARY_VALVE:18,
                                    ContourType.APEX_POINT:19,
                                    ContourType.RV_INSERT:20}
        for i in range(len(self.contour_type)):
            ctype[i] = type2int[self.contour_type[i]]
        mesh = io.Mesh(self.points_coordinates, {'vertex': np.arange(len(self.points_coordinates))[:,None]},
                point_data = {'ctype': ctype})
        return mesh

