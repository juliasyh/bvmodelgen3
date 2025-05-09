#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/11/16 15:55:30

@author: Javiera Jilberto Vallejos 
'''

import os

import numpy as np
import nibabel as nib
import meshio as io
import cheartio as chio

from tqdm import tqdm

from niftiutils import readFromNIFTI
from seg2contours import ViewSegFrame, SegSliceContour, add_apex, add_rv_apex, modify_sa_weights
from segutils import remove_base_nodes, writeResults
import segutils as su
import plotutils as pu
import valveutils as vu
import modelutils as mu
import slicealign as slicealign
from bvfitting import BiventricularModel, GPDataSet, ContourType

import pathlib
filepath=pathlib.Path(__file__).parent.resolve()

class PatientData:
    def __init__(self, img_paths, output_fldr):
        """
        Initializes the PatientData object.

        Parameters:
        img_paths (dict): A dictionary containing image paths with keys as identifiers.
        seg_paths (dict, optional): A dictionary containing segmentation paths with keys as identifiers. 
                                    If None, initializes with None for each key in img_paths.

        Attributes:
        img_paths (dict): A dictionary containing validated image paths.
        seg_paths (dict): A dictionary containing segmentation paths or None if not provided.
        output_fldr (str): The folder where output files will be saved.
        cmr_data (CMRSegData): An instance of the CMRSegData class containing segmentation data.
        """
        self.img_paths = self.check_nifti_extension(img_paths)
        self.output_fldr = output_fldr
        self.img2model_fldr = f'{self.output_fldr}/img2model/'
        self.seg_paths = None
        self.cmr_data = None

        # Initialize dummy template and list containing the meshes
        self.template = FittedTemplate(self.img2model_fldr, filemod='_mod2')
        self.surface_meshes = []
        self.volume_meshes = []


    @staticmethod
    def check_nifti_extension(img_paths):
        """
        Checks and validates the file paths for NIfTI images in the provided dictionary.

        This function iterates over the provided dictionary of image paths, checks if the paths
        have a '.nii' extension, and verifies their existence. If the path does not have a '.nii'
        extension, it checks for the existence of the file with '.nii.gz' or '.nii' extensions.
        If the file is found, it updates the path in the dictionary. If the file is not found,
        it raises a FileNotFoundError.

        Parameters:
        img_paths (dict): A dictionary where keys are view names and values are file paths to NIfTI images.

        Returns:
        dict: The updated dictionary with validated and corrected file paths.

        Raises:
        FileNotFoundError: If any of the specified files or their possible extensions do not exist.
        """
        for view, path in img_paths.items():
            if '.nii' in path:
                if os.path.exists(path):
                    continue
                else:
                    raise FileNotFoundError(f'File not found: {path}')
            if os.path.exists(f'{path}.nii.gz'):
                path = f'{path}.nii.gz'
            elif os.path.exists(f'{path}.nii'):
                path = f'{path}.nii'
            else:
                raise FileNotFoundError(f'File not found: {path}')
            img_paths[view] = path

        return img_paths
    

    def load_segmentations(self, seg_paths,
                          sa_labels={'lvbp': 3, 'lv': 2, 'rv': 1}, 
                          la_labels={'lvbp': 1, 'lv': 2, 'rv': 3}):
        
        print('Loading segmentations, using sa_labels = ', sa_labels, ' and la_labels = ', la_labels)
        self.seg_paths = self.check_nifti_extension(seg_paths)
        self.cmr_data = CMRSegData(self.img_paths, self.seg_paths, self.output_fldr, sa_labels, la_labels)
            
            

    def unet_generate_segmentations(self, list=None):
        """
        Labels images using NNUNet. If no list is provided, 
        it defaults to processing 'sa', 'la_2ch', 'la_4ch', and 'la_3ch' views.
        Args:
            list (list, optional): A list of view names to process. Defaults to 
            ['sa', 'la_2ch', 'la_4ch', 'la_3ch'].
        Returns:
            None
        """

        try:
            import CMRnn as nn
        except ImportError:
            raise ImportError('NNUNet is not installed. Please install it to use this function.')

        print('Generating segmentations using NNUNet...')
        if list is None:
            list = ['sa', 'la_2ch', 'la_4ch', 'la_3ch']
            
        seg_paths = {}
        for view in list:
            filepath = self.img_paths[view]
            working_fldr = os.path.dirname(filepath)

            imgs_fldr = nn.split_nifti(filepath)
            segs_fldr = nn.predict_nifti(imgs_fldr, view)
            nn.join_nifti(segs_fldr, view)
            nn.clean_fldr(working_fldr)

            print(f'Finished label prediction for {view} view.')
            seg_paths[view] = f'{working_fldr}/{view}_seg.nii.gz'

        return seg_paths


    def find_valves(self, valve_paths, visualize=False, load_from_nii=False, 
                    slices_2ch=[], slices_3ch=[], slices_4ch=[]) -> None:
        """
        Finds valve points in the segmentations.

        Args:
            valve_paths (dict): A dictionary containing paths to the valve segmentations.
            load_from_nii (bool, optional): Whether to load the valve points from NIfTI files. Defaults to False.

        Returns:
            None
        """
        self.valve_paths = valve_paths
        self.valve_data = CMRValveData(self.cmr_data.segs, self.valve_paths, f'{self.output_fldr}/img2model/')
        
        # Load or find valves
        if load_from_nii:
            self.valve_data.load_valves_from_nii(slices_2ch=slices_2ch, slices_3ch=slices_3ch, slices_4ch=slices_4ch)
        else:
            self.valve_data.find_valves(slices_2ch=slices_2ch, slices_3ch=slices_3ch, slices_4ch=slices_4ch)
        
        # Save valves
        self.valve_data.save_valves()

        if visualize:
            if len(slices_2ch) == 0:
                slice_2ch = 0
            else:
                slice_2ch = slices_2ch[0]
            if len(slices_3ch) == 0:
                slice_3ch = 0
            else:
                slice_3ch = slices_3ch[0]
            if len(slices_4ch) == 0:
                slice_4ch = 0
            else:
                slice_4ch = slices_4ch[0]
            self.valve_data.visualize_valves(self.cmr_data.imgs, 
                                             slices_to_plot={'la_2ch': slice_2ch, 
                                                             'la_3ch': slice_3ch, 
                                                             'la_4ch': slice_4ch})
            

    def generate_contours(self, min_base_length=15, visualize=True, which_frames=None, sa_min_weight=0.5):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)
        for frame in which_frames:
            frame_prefix = f'{self.img2model_fldr}/frame{frame}_'

            # Add valves to contours
            contours = self.cmr_data.all_frame_contours[frame]
            for view in self.cmr_data.segs.keys():
                if 'sa' in view: continue
                nslices = self.cmr_data.segs[view].shape[-2]
                for slice in range(nslices):
                    ijk_all_frames = self.valve_data.mv_points[view][slice]
                    if ijk_all_frames is None: continue
                    ijk = ijk_all_frames[frame]
                    ijk = np.column_stack((ijk, np.full(ijk.shape[0], slice)))
                    mv_points = self.cmr_data.apply_affine_to_pixcoords(ijk, view)

                    # Add to contour list
                    cont = SegSliceContour(mv_points, 'mv', slice, view)
                    contours.append(cont)
                    if '3ch' in view:
                        ijk_all_frames = self.valve_data.av_points[view][slice]
                        ijk = ijk_all_frames[frame]
                        ijk = np.column_stack((ijk, np.full(ijk.shape[0], slice)))
                        av_points = self.cmr_data.apply_affine_to_pixcoords(ijk, view)
                        # Add to contour list
                        cont = SegSliceContour(av_points, 'av', slice, view)
                        contours.append(cont)
                    if '4ch' in view:
                        ijk_all_frames = self.valve_data.tv_points[view][slice]
                        ijk = ijk_all_frames[frame]
                        ijk = np.column_stack((ijk, np.full(ijk.shape[0], slice)))
                        tv_points = self.cmr_data.apply_affine_to_pixcoords(ijk, view)
                        # Add to contour list
                        cont = SegSliceContour(tv_points, 'tv', slice, view)
                        contours.append(cont)

            # # Add apex
            add_apex(contours, self.cmr_data.segs)
            add_rv_apex(contours, self.cmr_data.segs)
            remove_base_nodes(contours, min_length=min_base_length)

            # Modify SA weights
            modify_sa_weights(contours, min_weight=sa_min_weight)

            # Write results
            writeResults(frame_prefix + 'contours.txt', contours)

            # Save .vtu
            mesh = pu.contours2vertex(contours)
            io.write(f'{frame_prefix}contours.vtu', mesh)

            fig = pu.plot_contours(contours, background=True)
            if visualize:
                fig.show()
            pu.save_figure(frame_prefix + 'contours.html', fig)
        

    def fit_templates(self, which_frames=None, weight_GP=1, low_smoothing_weight=10, 
                      transmural_weight=20, rv_thickness=3, reuse_control_points=False, make_vol_mesh=False):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        self.surface_meshes = []
        load_control_points = None

        if make_vol_mesh:
            self.volume_meshes = []

        for frame in range(self.cmr_data.nframes):
            if frame not in which_frames:
                self.surface_meshes.append(None)
                continue
            print(f'Fitting template for frame {frame}...')
            frame_prefix = f'{self.img2model_fldr}/frame{frame}_'

            filename = frame_prefix + 'contours.txt'
            if reuse_control_points and frame > 0:
                load_control_points = f'{self.img2model_fldr}/frame{frame-1}_control_points.npy'

            template = FittedTemplate(frame_prefix, filemod='_mod2')
            template.load_dataset(filename, weight_GP, load_control_points=load_control_points)

            template.fit_template(weight_GP, low_smoothing_weight, transmural_weight, rv_thickness)
            surface_mesh = template.bvmodel.get_template_mesh()

            if make_vol_mesh:
                vol_mesh = template.make_volumetric_mesh()
                self.volume_meshes.append(vol_mesh)

            self.surface_meshes.append(surface_mesh)

            print('-------------------------------------------\n')

        self.template = template

        if make_vol_mesh:
            return self.surface_meshes, self.volume_meshes
        return self.surface_meshes


    def smooth_surfaces_in_time(self, surfaces, which_frames=None):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        # Only do this if the whole cycle is considered
        if len(which_frames) != self.cmr_data.nframes:
            print('Smoothing BV surfaces in time is only available for the whole cycle. \n'
                  'Returning without smoothing.')
            return surfaces
 
        print('Smoothing BV surfaces in time...')

        # Smooth the surfaces
        smooth_surfaces = mu.fourier_time_smoothing(surfaces)
        self.surface_meshes = smooth_surfaces

        return self.surface_meshes


    def correct_surfaces_by_volumes(self, surfaces, which_frames=None):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        # Only do this if the whole cycle is considered
        if len(which_frames) != self.cmr_data.nframes:
            print('Correcting BV surfaces by volumes is only available for the whole cycle. \n'
                'Returning without correcting.')
            return surfaces
        
        print('Correcting BV surfaces by volumes...')
        labels = self.template.labels
        new_surfaces = mu.volume_match_correction(surfaces, labels)
        self.surface_meshes = new_surfaces

        return self.surface_meshes


    def load_surfaces(self, which_frames=None):
        """
        Loads the surfaces from a file.

        Args:
            path (str): The path to the file containing the surfaces.

        Returns:
            list: A list of loaded surfaces.
        """
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        surfaces = []
        for frame in range(self.cmr_data.nframes):
            if frame not in which_frames:
                surfaces.append(None)
                continue
            surface_mesh = io.read(f'{self.img2model_fldr}/frame{frame}_template.vtu')
            surfaces.append(surface_mesh)
            self.surface_meshes.append(surface_mesh)

        return surfaces
    

    def save_bv_surfaces(self, surfaces, mesh_subdivisions=0, prefix='', which_frames=None, save_volumes=False):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        # Grab infor from the template
        labels = self.template.labels
        bv_regions = self.template.bv_labels
        valve_regions = self.template.valve_labels
        septum_regions = self.template.septum_labels

        for frame in range(self.cmr_data.nframes):
            if frame not in which_frames:
                continue
            surface_mesh = surfaces[frame]

            # Extract bvmesh
            bvmesh = mu.extract_subregion(surface_mesh, labels, bv_regions)

            # Extract valve mesh
            valve_mesh = mu.extract_subregion(surface_mesh, labels, valve_regions)

            # Extract septum mesh
            septum_mesh = mu.extract_subregion(surface_mesh, labels, septum_regions)

            # Subdivide the mesh
            if mesh_subdivisions > 0:
                bvmesh = mu.subdivide_mesh(bvmesh, mesh_subdivisions)
                valve_mesh = mu.subdivide_mesh(valve_mesh, mesh_subdivisions)
                septum_mesh = mu.subdivide_mesh(septum_mesh, mesh_subdivisions)

            io.write(f'{self.img2model_fldr}/frame{frame}_{prefix}template.vtu', surface_mesh)
            io.write(f'{self.img2model_fldr}/frame{frame}_{prefix}bv_surface.stl', bvmesh)
            io.write(f'{self.img2model_fldr}/frame{frame}_{prefix}valve_surfaces.stl', valve_mesh)
            io.write(f'{self.img2model_fldr}/frame{frame}_{prefix}septum_surface.stl', septum_mesh)

            if save_volumes:
                vol_mesh = self.volume_meshes[frame]
                io.write(f'{self.img2model_fldr}/frame{frame}_bv_volume.vtu', vol_mesh)


    def calculate_chamber_volumes(self, surfaces, which_frames=None):
        #TODO add option to enter a bv surface mesh instead of a template mesh
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        lv_volume = np.zeros(self.cmr_data.nframes)
        rv_volume = np.zeros(self.cmr_data.nframes)

        labels = self.template.labels
        for frame in range(self.cmr_data.nframes):
            if frame not in which_frames:
                lv_volume[frame] = None
                rv_volume[frame] = None
                continue
            surface_mesh = surfaces[frame]
            xyz = surface_mesh.points
            ien = surface_mesh.cells[0].data

            lv_volume[frame], rv_volume[frame] = mu.calculate_chamber_volumes(xyz, ien, labels)

        which_frames = np.array(which_frames)

        # Save the volumes
        lv_volume = np.array(lv_volume)[which_frames]
        save = np.column_stack((which_frames, lv_volume))
        chio.write_dfile(f'{self.output_fldr}/lv_volumes.txt', save)
        
        rv_volume = np.array(rv_volume)[which_frames]
        save = np.column_stack((which_frames, rv_volume))
        chio.write_dfile(f'{self.output_fldr}/rv_volumes.txt', save)

        return lv_volume, rv_volume
        

    def calculate_wall_volumes(self, surfaces, which_frames=None):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.cmr_data.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        lv_wall_volume = np.zeros(self.cmr_data.nframes)
        rv_wall_volume = np.zeros(self.cmr_data.nframes)

        labels = self.template.labels
        for frame in range(self.cmr_data.nframes):
            if frame not in which_frames:
                lv_wall_volume[frame] = None
                rv_wall_volume[frame] = None
                continue
            surface_mesh = surfaces[frame]
            xyz = surface_mesh.points
            ien = surface_mesh.cells[0].data

            lv_wall_volume[frame], rv_wall_volume[frame] = mu.calculate_wall_volumes(xyz, ien, labels)

        which_frames = np.array(which_frames)

        # Save the volumes
        lv_volume = np.array(lv_wall_volume)[which_frames]
        save = np.column_stack((which_frames, lv_volume))
        chio.write_dfile(f'{self.output_fldr}/lv_wall_volumes.txt', save)
        
        rv_volume = np.array(rv_wall_volume)[which_frames]
        save = np.column_stack((which_frames, rv_volume))
        chio.write_dfile(f'{self.output_fldr}/rv_wall_volumes.txt', save)

        return lv_volume, rv_volume




class CMRSegData:
    def __init__(self, img_paths, seg_paths, output_fldr,
                          sa_labels={'lvbp': 3, 'lv': 2, 'rv': 1}, 
                          la_labels={'lvbp': 1, 'lv': 2, 'rv': 3}):
        """
        Initializes the CMRSegData object.

        Parameters:
        img_paths (dict): A dictionary containing image paths with keys as identifiers.
        seg_paths (dict): A dictionary containing segmentation paths with keys as identifiers.
        output_fldr (str): The folder where output files will be saved.
        sa_labels (dict): A dictionary containing labels for short-axis views.
        la_labels (dict): A dictionary containing labels for long-axis views.

        Attributes:
        img_paths (dict): A dictionary containing validated image paths.
        seg_paths (dict): A dictionary containing validated segmentation paths.
        output_fldr (str): The folder where output files will be saved.
        sa_labels (dict): A dictionary containing labels for short-axis views.
        la_labels (dict): A dictionary containing labels for long-axis views.
        imgs (dict): A dictionary containing ViewImgData objects for each view.
        segs (dict): A dictionary containing ViewSegData objects for each view.
        nframes (int): The number of frames in the segmentation data.
        img2model_fldr (str): The folder where intermediate steps will be saved.
        """
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.output_fldr = output_fldr
        self.sa_labels = sa_labels
        self.la_labels = la_labels

        # Read images
        self.imgs = {}
        for view, path in img_paths.items():
            (data, affine, pixdim) = readFromNIFTI(path)
            self.imgs[view] = ViewImgData(view, data, affine, pixdim, path)

        # Read segmentations
        self.segs = {}
        for view, path in self.seg_paths.items():
            (data, affine, pixdim) = readFromNIFTI(path)

            if 'sa' in view:
                labels = self.sa_labels
            else:
                labels = self.la_labels
            self.segs[view] = ViewSegData(view, data, affine, pixdim, labels, path)

        # Calculate number of frames
        self.nframes = self.get_frame_number(self.segs)

        # Create a folder to save the intermediate steps
        self.img2model_fldr = f'{self.output_fldr}/img2model/'
        if not os.path.exists(self.img2model_fldr):
            os.makedirs(self.img2model_fldr)

        # Intermediate lists
        self.all_frame_slices = None
        self.all_frame_translations = None
        self.all_frame_contours = None


    @staticmethod
    def get_frame_number(segs):
        # Check that all segmentations have the same number of frames
        frames = [seg.shape[-1] for seg in segs.values()]
        if len(set(frames)) > 1:
            frame_info = ', '.join([f'{view}: {seg.shape[-1]}' for view, seg in segs.items()])
            raise ValueError(f'Segmentations have different number of frames. Frame counts: {frame_info}')
        return frames[0]
    

    def clean_segmentations(self, min_ecc=0.6):

        for view in self.segs.keys():
            seg = self.segs[view]
            seg.clean_data(min_ecc=min_ecc)


    def extract_slices(self, visualize=True, which_frames=[0]):
        print('Extracting slices...')

        all_frame_slices = []
        for frame in tqdm(range(self.nframes)):
            if frame not in which_frames:
                all_frame_slices.append([])
                continue
            frame_prefix = f'{self.img2model_fldr}/frame{frame}_'

            # Extract slices from all the segmentations
            slices =[]
            for view in self.segs.keys():
                # Grab frame data, affine, and zooms
                frame_data = self.segs[view].data[:, :, :, frame]
                affine = self.segs[view].affine
                zooms = self.segs[view].pixdim

                # Create segmntation object
                if 'sa' in view:
                    labels = self.sa_labels
                else:
                    labels = self.la_labels

                view_seg = ViewSegFrame(view, frame_data, affine, zooms, labels, frame)
                slices += view_seg.extract_slices(defseptum=True)
            all_frame_slices.append(slices)

            fig = pu.plot_slices(slices)
            if visualize:
                fig.show()
            pu.save_figure(frame_prefix + 'initial_contours.html', fig)

        return all_frame_slices
    

    def align_frame_slices(self, slices, frame_prefix, nslices, translation_files_prefix=None, method=2, visualize=False):
        if translation_files_prefix is None:
            # Compute alignment
            print('Calculating alignment using Sinclair algorithm...')
            slicealign.find_SA_initial_guess(slices)
            if method == 2:
                slicealign.optimize_stack_translation2(slices, nit=100)
            elif method == 3:
                slicealign.optimize_stack_translation3(slices, nit=100)
            translations = slicealign.save_translations(frame_prefix, nslices, slices)

        else:
            translations = {}
            found = 0
            for view in  nslices.keys():
                try:
                    translations[view] = np.load(f'{translation_files_prefix}{view.lower()}_translations.npy')
                    found += 1
                except:
                    print('Translation file for ' + view + ' not found.')
                    continue
                if 'frame0' in frame_prefix:
                    print('Loading translation file for ' + view + '...')

            if found == 0:
                # Compute alignment
                print('No translation file found, calculating alignment using Sinclair algorithm...')
                slicealign.find_SA_initial_guess(slices)
                if method == 2:
                    slicealign.optimize_stack_translation2(slices, nit=100)
                elif method == 3:
                    slicealign.optimize_stack_translation3(slices, nit=100)
                translations = slicealign.save_translations(frame_prefix, nslices, slices)


            new_slices = []
            for slc in slices:
                view = slc.view
                trans = translations[view][slc.slice_number]
                slc.accumulated_translation = trans
                if not np.all(trans == 0):
                    new_slices.append(slc)
            slices = new_slices

        fig = pu.plot_slices(slices)
        if visualize:
            fig.show()
        pu.save_figure(frame_prefix + 'aligned_contours.html', fig)

        return slices, translations
    

    def align_slices(self, all_frame_slices, visualize=False, which_frames=[0]):
        print('Aligning slices...')

        frame0_prefix = f'{self.img2model_fldr}/frame{0}_'
        all_frame_slices_aligned = []
        all_frame_translations = self.all_frame_translations
        for frame in tqdm(range(self.nframes)):
            if frame not in which_frames:
                all_frame_slices_aligned.append([])
                continue
            frame_prefix = f'{self.img2model_fldr}/frame{frame}_'

            # Grab length of data
            nslices = {}
            for view in self.segs.keys():
                nslices[view] = self.segs[view].shape[-1]

            # Align slices
            slices, translations = self.align_frame_slices(all_frame_slices[frame], frame_prefix, nslices, 
                                                           translation_files_prefix=frame0_prefix, visualize=visualize)

            all_frame_slices_aligned.append(slices)
            all_frame_translations[frame] = translations

        return all_frame_slices_aligned, all_frame_translations
    
    def generate_null_translations(self):
        all_frame_translations = []
        for frame in tqdm(range(self.nframes)):
            translations = {}
            for view in self.segs.keys():
                translations[view] = np.zeros([self.segs[view].shape[-1], 2])
            all_frame_translations.append(translations)
        return all_frame_translations


    def generate_contours(self, all_frame_slices, downsample=3, visualize=True, which_frames=[0]):
        print('Generating contours...')

        all_frame_contours = []
        for frame in tqdm(range(self.nframes)):
            if frame not in which_frames:
                all_frame_contours.append([])
                continue
            frame_prefix = f'{self.img2model_fldr}/frame{frame}_'
            contours = []

            slices = all_frame_slices[frame]
            for slc in slices:
                contours += slc.tocontours(downsample)

            # Visualize
            fig = pu.plot_contours(contours, background=True)
            pu.save_figure(self.img2model_fldr + 'contours.html', fig)
            if visualize:
                fig.show()

            self.vertex_contours = pu.contours2vertex(contours)
            io.write(f'{frame_prefix}contours.vtu', self.vertex_contours)
            all_frame_contours.append(contours)

        return all_frame_contours
    
    def contour_sanity_check(self, contours, which_frames=None):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        else:
            which_frames = list(which_frames)

        for frame in range(len(contours)):
            if frame not in which_frames: continue
            frame_ctrs = contours[frame]
            la_contours = []
            sa_contours = []
            for ctr in frame_ctrs:
                if 'la' in ctr.view:
                    la_contours.append(ctr)
                else:
                    sa_contours.append(ctr)

            assert len(sa_contours) > 0, 'No SA contours found in frame ' + str(frame)
            assert len(la_contours) > 0, 'No LA contours found in frame ' + str(frame)


    def extract_contours(self, visualize=True, which_frames=None, align=True):
        # Deal with which_frames
        if which_frames is None:
            which_frames = range(self.nframes)
        elif isinstance(which_frames, int):
            which_frames = [which_frames]
        elif isinstance(which_frames, list):
            which_frames = np.array(which_frames)
        else:
            which_frames = list(which_frames)

        # Extract slices
        self.all_frame_slices = self.extract_slices(visualize=visualize, which_frames=which_frames)

        # Align slices
        self.all_frame_translations = self.generate_null_translations()    
        if align:
            self.all_frame_slices, self.all_frame_translations = self.align_slices(self.all_frame_slices, which_frames=which_frames)
            

        # Generate contours
        self.all_frame_contours = self.generate_contours(self.all_frame_slices, visualize=visualize, 
                                                         which_frames=which_frames)
        
        self.contour_sanity_check(self.all_frame_contours, which_frames)

    
    def apply_affine_to_pixcoords(self, ijk, view, translation_frame=0):
        # load respective translations and apply it to ijk
        translations = self.all_frame_translations[translation_frame][view]

        for i in range(len(ijk)):
            ijk[i, 0:2] += translations[int(ijk[i, 2])]
    
        affine = self.segs[view].affine
        xyz = nib.affines.apply_affine(affine, ijk)

        return xyz

    def load_rib_diaphragm(self, filename, labels={'rib': 1, 'diaphragm': 2}, frame=0):
        seg, affine, pixdim = readFromNIFTI(filename)
    
        frame_data = seg[:, :, :, frame]
        
        # Grab points
        ribs_ijk = np.column_stack(np.where(frame_data == labels['rib'])).astype(float)
        diaphragm_ijk = np.column_stack(np.where(frame_data == labels['diaphragm'])).astype(float)

        # load respective translations and apply it to ijk
        translations = self.all_frame_translations[frame]['sa']

        # print(ribs_ijk[:10])
        # print(nib.affines.apply_affine(affine, ribs_ijk)[:10])
        for i in range(len(ribs_ijk)):
            ribs_ijk[i, 0:2] += translations[int(ribs_ijk[i, 2])]
        for i in range(len(diaphragm_ijk)):
            diaphragm_ijk[i, 0:2] += translations[int(diaphragm_ijk[i, 2])]
        # print(ribs_ijk[:10])

        # Transform points
        ribs_xyz = nib.affines.apply_affine(affine, ribs_ijk)
        diaphragm_xyz = nib.affines.apply_affine(affine, diaphragm_ijk)
        print(ribs_xyz[:10])

        return ribs_xyz, diaphragm_xyz


class ViewImgData:
    def __init__(self, view, data, affine, pixdim, path):
        self.view = view
        self.data = data
        self.affine = affine
        self.pixdim = pixdim
        self.shape = data.shape
        self.path = path
        

class ViewSegData:
    def __init__(self, view, data, affine, pixdim, labels, path):
        self.view = view
        self.affine = affine
        self.pixdim = pixdim
        self.labels = labels
        self.path = path
        self.data = data

        self.shape = data.shape

    def inverse_transform(self, point):
        A, t = nib.affines.to_matvec(self.affine)
        ijk = np.linalg.solve(A, point-t)
        ijk = np.floor(ijk)
        return ijk
    

    def clean_data(self, min_ecc=0.6):
        print(f'Cleaning {self.view} segmentation...')
        new_data = np.zeros_like(self.data)

        # Loop over slices
        for frame in range(self.data.shape[3]):
            for i in range(self.data.shape[2]):
                slc = self.data[:,:,i,frame]

                # Get segmentations
                lv = np.isclose(slc, self.labels['lv'])
                rv = np.isclose(slc, self.labels['rv']) 
                if self.view == 'la_2ch': rv[:] = 0
                lvbp = np.isclose(slc, self.labels['lvbp'])

                mask = lv+rv+lvbp
                
                if np.all(mask == 0):   # No segmentations
                    continue

                # Clean each segmentation
                min_area = np.sum(mask) * 0.01
                rv = su.remove_holes_islands(rv, irregMaxSize=min_area).astype(bool)
                lvbp = su.remove_holes_islands(lvbp, irregMaxSize=min_area).astype(bool)

                if 'sa' in self.view:
                    lv = su.remove_holes_islands(lv, irregMaxSize=min_area) - lvbp
                else:
                    lv = su.remove_holes_islands(lv+lvbp, irregMaxSize=min_area) - lvbp

                if np.min(lv) < 0:
                    print(f'WARNING: The LV in {self.view}, frame {frame}, slice {i} is not closed, deleting slice data')
                    lv[:] = 0
                    lvbp[:] = 0
                    rv[:] = 0

                # Check that there is only one object for each segmentation
                rv_n = su.get_number_of_objects(rv)
                lvbp_n = su.get_number_of_objects(lvbp)
                lv_n = su.get_number_of_objects(lv)

                if rv_n > 1: 
                    print(f'WARNING: More than one RV object in slice {i} frame {frame}, deleting slice data')
                    rv[:] = 0
                if (lvbp_n > 1) or (lv_n > 1): 
                    if 'la' in self.view:
                        raise ValueError(f'ERROR: More than one LV or LVBP object in slice {i} frame {frame}')
                    else:
                        print(f'WARNING: More than one LV or LVBP object in slice {i} frame {frame}, deleting slice data')
                        lvbp[:] = 0
                        lv[:] = 0
                        rv[:] = 0
                    
                if not np.all(lv == 0) and (self.view == 'sa'):
                    # Check that the LV is somewhat round
                    try:
                        lv_ecc = su.get_mask_eccentricity(lv)
                    except:
                        from matplotlib import pyplot as plt
                        plt.imshow(lv)
                        plt.show()
                    if lv_ecc > min_ecc:
                        print(f'WARNING: The LV eccentricity > {min_ecc} in {self.view}, frame {frame}, slice {i}, deleting slice data')
                        lv[:] = 0
                        lvbp[:] = 0
                        rv[:] = 0

                # Save segmentations
                lv = lv.astype(int)
                rv = rv.astype(int)
                lvbp = lvbp.astype(int)

                new_data[:,:,i,frame] = lv*self.labels['lv'] + rv*self.labels['rv'] + lvbp*self.labels['lvbp']
                

            # Secondary check for SA. The SA slices with data should be contigous (no gaps)
            if 'sa' in self.view:
                sum_data = np.sum(new_data[:,:,:,frame], axis=(0,1))
                nonzero = np.where(sum_data != 0)[0]
                diff = np.diff(nonzero)
                lims = np.where(diff > 1)[0]
                if len(lims) > 0: # There are non-contiguous slices
                    nsets = len(lims) + 1
                    lims = np.concatenate(([0], lims+1, [len(nonzero)]))

                    # Grab the longest one
                    lengths = []
                    for i in range(nsets):
                        lengths.append(lims[i+1] - lims[i])

                    idx = np.argmax(lengths)
                    start = lims[idx]
                    end = lims[idx+1]

                    # Set all the other slices to 0
                    nonzero = nonzero[start:end]
                
                for i in range(new_data.shape[2]):
                    if i not in nonzero:
                        new_data[:,:,i,frame] = 0

        self.data = new_data

            
class CMRValveData:
    def __init__(self, segs, valve_paths, output_fldr):
        self.segs = segs
        self.valve_paths = valve_paths
        self.output_fldr = output_fldr
        
        # Initialize dictionaries
        self.mv_points = {}
        self.mv_centroids = {}
        self.av_points = {}
        self.av_centroids = {}
        self.tv_points = {}
        self.tv_centroids = {}
        self.initialize_valve_dicts()


    def initialize_valve_dicts(self):
        for view in self.segs.keys():
            if 'sa' in view: continue
            nslices = self.segs[view].shape[-2]

            self.mv_points[view] = {}
            self.mv_centroids[view] = {}
            for i in range(nslices):
                self.mv_points[view][i] = None
                self.mv_centroids[view][i] = None

            if '3ch' in view:
                self.av_points[view] = {}
                self.av_centroids[view] = {}
                for i in range(nslices):
                    self.av_points[view][i] = None
                    self.av_centroids[view][i] = None

            elif '4ch' in view:
                self.tv_points[view] = {}
                self.tv_centroids[view] = {}
                for i in range(nslices):
                    self.tv_points[view][i] = None
                    self.tv_centroids[view][i] = None


    def find_valves(self, slices_2ch=[], slices_3ch=[], slices_4ch=[]):
        print('Finding valves...')
        # TODO need to request there is a valve seg for all the slices that we are using
        # TODO make sure TV second point is the septal one
        # Note that MV points are returned such that the second point is the septal one
        # 2CH, automatic, no need to load from nii
        view = 'la_2ch'
        if view in self.valve_paths.keys():
            print('Finding MV valve in 2CH view...')
            seg = self.segs[view]
            nslices = seg.shape[-2]
            if len(slices_2ch) == 0:
                slices_2ch = range(nslices)

            mv_seg_points = vu.load_valve_nii(self.valve_paths[view], view)

            for slice in range(nslices):
                if nslices > 1:
                    if slice not in slices_2ch: continue
                mv_seg_points_slice = mv_seg_points[mv_seg_points[:,2]==slice][:,0:2]
                self.mv_points[view][slice], self.mv_centroids[view][slice] = vu.get_2ch_valve_points(seg, mv_seg_points_slice, slice=slice) 


        # For the 3CH we only use one slice
        view = 'la_3ch'
        if view in self.valve_paths.keys():
            print('Finding AV and MV valve in 3CH view...')
            seg = self.segs[view]
            nslices = seg.shape[-2]
            if len(slices_3ch) == 0:
                slices_3ch = range(nslices)

            mv_seg_points, av_seg_points = vu.load_valve_nii(self.valve_paths[view], view)

            for slice in range(nslices):
                if nslices > 1:
                    if slice not in slices_3ch: continue
                mv_seg_points_slice = mv_seg_points[mv_seg_points[:,2]==slice][:,0:2]
                av_seg_points_slice = av_seg_points[av_seg_points[:,2]==slice][:,0:2]

                # Sanity checks
                if nslices > 1:
                    if len(mv_seg_points_slice) == 0:
                        print('WARNING: No MV points found in 3CH view, slice', slice)
                        continue
                    if len(av_seg_points_slice) == 0:
                        print('WARNING: No AV points found in 3CH view, slice', slice)
                        continue
                else:
                    assert len(mv_seg_points_slice) > 0, 'No MV points found in 3CH view'
                    assert len(av_seg_points_slice) > 0, 'No AV points found in 3CH view'

                output = vu.get_3ch_valve_points(seg, slice=slice, mv_seg_points=mv_seg_points_slice, 
                                                                   av_seg_points=av_seg_points_slice)
                self.mv_points[view][slice] = output[0]
                self.mv_centroids[view][slice] = output[1]
                self.av_points[view][slice] = output[2]
                self.av_centroids[view][slice] = output[3]

        # 4CH
        view = 'la_4ch'
        if view in self.valve_paths.keys():
            print('Finding MV and TV valve in 4CH view...')
            seg = self.segs[view]
            nslices = seg.shape[-2]
            if len(slices_4ch) == 0:
                slices_4ch = range(nslices)
            mv_seg_points, tv_seg_points = vu.load_valve_nii(self.valve_paths[view], view)
            

            for slice in range(nslices):
                if nslices > 1:
                    if slice not in slices_4ch: continue
                
                mv_seg_points_slice = mv_seg_points[mv_seg_points[:,2]==slice][:,0:2]
                tv_seg_points_slice = tv_seg_points[tv_seg_points[:,2]==slice][:,0:2]

                # Sanity checks
                if nslices > 1:
                    if len(mv_seg_points_slice) == 0:
                        print('WARNING: No MV points found in 4CH view, slice', slice)
                        continue
                    if len(tv_seg_points_slice) == 0:
                        print('WARNING: No TV points found in 4CH view, slice', slice)
                        continue
                else:
                    assert len(mv_seg_points_slice) > 0, 'No MV points found in 4CH view'
                    assert len(tv_seg_points_slice) > 0, 'No TV points found in 4CH view'

                output = vu.get_4ch_valve_points(seg, mv_seg_points_slice, tv_seg_points_slice, slice)
                self.mv_points[view][slice] = output[0] 
                self.mv_centroids[view][slice] = output[1]
                self.tv_points[view][slice] = output[2] 
                self.tv_centroids[view][slice] = output[3]


    def load_valves_from_nii(self,slices_2ch=[], slices_3ch=[], slices_4ch=[]):
        print('Loading valves from NIfTI files...')
        nframes=self.segs['la_2ch'].shape[-1]

        view = 'la_2ch'
        if view in self.valve_paths.keys():
            mv_pnt_arr=np.zeros([nframes,2,2])
            mv_cent_arr=np.zeros([nframes,2])
            for iframe in range(nframes):
                seg = self.segs[view]
                nslices = seg.shape[-2]
                if len(slices_2ch) == 0:
                    slices_2ch = range(nslices)

                mv_seg_points = vu.load_valve_nii(self.valve_paths[view], view,frame=iframe)
                
                for islice in range(nslices):
                    if nslices > 1:
                        if islice not in slices_2ch: continue
                    mv_pnt_arr[iframe,:,:]=mv_seg_points[mv_seg_points[:,2]==islice][:,0:2]
                    mv_cent_arr[iframe,:]=np.mean(mv_pnt_arr[iframe,:,:], axis=0)
            
            self.mv_points[view][islice] = mv_pnt_arr
            self.mv_centroids[view][islice] = mv_cent_arr

        # For the 3CH we only use one slice
        view = 'la_3ch'
        if view in self.valve_paths.keys():
            mv_pnt_arr=np.zeros([nframes,2,2])
            mv_cent_arr=np.zeros([nframes,2])
            av_pnt_arr=np.zeros([nframes,2,2])
            av_cent_arr=np.zeros([nframes,2])
            for iframe in range(nframes):
                seg = self.segs[view]
                nslices = seg.shape[-2]
                if len(slices_3ch) == 0:
                    slices_3ch = range(nslices)


                mv_seg_points, av_seg_points = vu.load_valve_nii(self.valve_paths[view], view,frame=iframe)

                for islice in range(nslices):
                    if nslices > 1:
                        if islice not in slices_2ch: continue
                    mv_pnt_arr[iframe,:,:]=mv_seg_points[mv_seg_points[:,2]==islice][:,0:2]
                    mv_cent_arr[iframe,:]=np.mean(mv_pnt_arr[iframe,:,:], axis=0)
                    av_pnt_arr[iframe,:,:]=av_seg_points[av_seg_points[:,2]==islice][:,0:2]
                    av_cent_arr[iframe,:]=np.mean(av_pnt_arr[iframe,:,:], axis=0)

            self.mv_points[view][islice] = mv_pnt_arr
            self.mv_centroids[view][islice] = mv_cent_arr
            self.av_points[view][islice] = av_pnt_arr
            self.av_centroids[view][islice] = av_cent_arr

        # 4CH
        view = 'la_4ch'
        if view in self.valve_paths.keys():
            mv_pnt_arr=np.zeros([nframes,2,2])
            mv_cent_arr=np.zeros([nframes,2])
            tv_pnt_arr=np.zeros([nframes,2,2])
            tv_cent_arr=np.zeros([nframes,2])
            for iframe in range(nframes):
                seg = self.segs[view]
                nslices = seg.shape[-2]
                if len(slices_4ch) == 0:
                    slices_4ch = range(nslices)
                mv_seg_points, tv_seg_points = vu.load_valve_nii(self.valve_paths[view], view,frame=iframe)
                
                for islice in range(nslices):
                    if nslices > 1:
                        if islice not in slices_2ch: continue
                    mv_pnt_arr[iframe,:,:]=mv_seg_points[mv_seg_points[:,2]==islice][:,0:2]
                    mv_cent_arr[iframe,:]=np.mean(mv_pnt_arr[iframe,:,:], axis=0)
                    tv_pnt_arr[iframe,:,:]=tv_seg_points[tv_seg_points[:,2]==islice][:,0:2]
                    tv_cent_arr[iframe,:]=np.mean(tv_pnt_arr[iframe,:,:], axis=0)


            self.mv_points[view][islice] = mv_pnt_arr
            self.mv_centroids[view][islice] = mv_cent_arr
            self.tv_points[view][islice] = tv_pnt_arr
            self.tv_centroids[view][islice] = tv_cent_arr

    
    def visualize_valves(self, imgs, slices_to_plot = {'la_2ch': 0, 'la_3ch': 0, 'la_4ch': 0}):

        for view in slices_to_plot.keys():
            valve_points = {'mv': self.mv_points[view]}
            valve_centroids = {'mv': self.mv_centroids[view]}
            if '3ch' in view:
                valve_points['av'] = self.av_points[view]
                valve_centroids['av'] = self.av_centroids[view]
            elif '4ch' in view:
                valve_points['tv'] = self.tv_points[view]
                valve_centroids['tv'] = self.tv_centroids[view]
                    
            vu.plot_valve_movement(imgs[view], self.segs[view], slice=slices_to_plot[view], 
                                valve_points=valve_points,
                                valve_centroids=valve_centroids)

    def save_valves(self,labels = {'mv': 1, 'tv': 2, 'av': 3, 'pv': 4}):
        for view in self.segs.keys():
            if 'sa' in view: continue
            # Old numpy saves
            np.save(f'{self.output_fldr}/{view}_mv_points.npy', self.mv_points[view])
            np.save(f'{self.output_fldr}/{view}_mv_centroids.npy', self.mv_centroids[view])

            img=self.segs[view]
            data=np.zeros_like(img.data)

            n_timesteps=self.segs[view].shape[-1]
            nslices=self.segs[view].shape[-2]
                
            for it in range(n_timesteps):
                for islc in range(nslices):
                    if self.mv_points[view][islc] is None: continue
                    mv=np.array(self.mv_points[view][islc].tolist()) 
                    p1x,p1y=mv[it,0,:]
                    p2x,p2y=mv[it,1,:]
                    data[int(p1x),int(p1y),islc,it]=labels['mv']
                    data[int(p2x),int(p2y),islc,it]=labels['mv']

            if '3ch' in view:
                for it in range(n_timesteps):
                    for islc in range(nslices):
                        if self.av_points[view][islc] is None: continue
                        av=np.array(self.av_points[view][islc].tolist()) 
                        p1x,p1y=av[it,0,:]
                        p2x,p2y=av[it,1,:]
                        data[int(p1x),int(p1y),islc,it]=labels['av']
                        data[int(p2x),int(p2y),islc,it]=labels['av']

                # Old numpy saves
                np.save(f'{self.output_fldr}/{view}_av_points.npy', self.av_points[view])
                np.save(f'{self.output_fldr}/{view}_av_centroids.npy', self.av_centroids[view])

            elif '4ch' in view:
                for it in range(n_timesteps):
                    for islc in range(nslices):
                        if self.tv_points[view][islc] is None: continue
                        tv=np.array(self.tv_points[view][islc].tolist()) 
                        p1x,p1y=tv[it,0,:]
                        p2x,p2y=tv[it,1,:]
                        data[int(p1x),int(p1y),islc,it]=labels['tv']
                        data[int(p2x),int(p2y),islc,it]=labels['tv']

                # Old numpy saves
                np.save(f'{self.output_fldr}/{view}_tv_points.npy', self.tv_points[view])
                np.save(f'{self.output_fldr}/{view}_tv_centroids.npy', self.tv_centroids[view])

            combined_img = nib.Nifti1Image(data, img.affine)
            outfile=f'{self.output_fldr}/{view}_valves_full.nii.gz'
            nib.save(combined_img, outfile)
            print('Saving valves...')


class FittedTemplate:
    template_fitting_weights = {'apex_endo': 2, 'apex_epi': 2, 'mv': 1., 'tv': 1, 'av': 1.5, 'pv': 1,
                                'mv_phantom': 2, 'tv_phantom': 1., 'av_phantom': 2., 'pv_phantom': 1,
                                'rv_insert': 1.5,
                                'la_rv_endo': 3, 'la_rv_epi': 2, 'la_lv_endo': 2, 'la_lv_epi': 1,
                                'sa_lv_epi': 1, 'sa_lv_endo': 2, 'sa_rv_endo': 1, 'sa_rv_epi': 1 }
    

    def __init__(self, out_prefix, filemod='_mod'):
        self.out_prefix = out_prefix
        self.load_control_points = None

        # Loads biventricular control_mesh
        model_path = f"{filepath}/bvfitting/template" # folder of the control mesh
        self.bvmodel = BiventricularModel(model_path, filemod=filemod)
        self.labels = self.bvmodel.surfs
        self.patches = {'lv_endo': 0,
                        'lv_epi': 8,
                        'rv_endo': 2,
                        'rv_epi': 3,
                        'rv_septum': 1,
                        'rvlv': 9,
                        'mv': 12,
                        'tv': 13,
                        'av': 10,
                        'pv': 11,}
        
        if filemod == '_mod2':
            self.patches['rvlv'] = 14

        self.dataset = None

        # Initialize surface meshes
        self.bvmesh = None
        self.valve_mesh = None
        self.septum_mesh = None
        self.template_mesh = None

        # List of labels for each region of interest
        self.bv_labels = [0,8,2,3,1,9,10,11,12,13]
        if filemod == '_mod2':
            self.bv_labels = [0,8,2,3,1,14,10,11,12,13]
        self.valve_labels = [12,13,10,11]
        self.septum_labels = [1]
            

    def load_dataset(self, filename, weight_GP, load_control_points=None):
        # Filename containing guide points (from contours/masks)
        self.dataset = GPDataSet(filename)
        self.load_control_points = None

        # Load position
        if load_control_points:
            self.bvmodel.control_mesh = np.load(load_control_points)
            self.bvmodel.et_pos = np.linalg.multi_dot([self.bvmodel.matrix,
                                                        self.bvmodel.control_mesh])

        else:
            # Procrustes alignment
            self.bvmodel.update_pose_and_scale(self.dataset)

            # perform a stiff fit
            displacement, err = self.bvmodel.lls_fit_model(weight_GP,self.dataset,1e10)
            self.bvmodel.control_mesh = np.add(self.bvmodel.control_mesh,
                                                    displacement)
            self.bvmodel.et_pos = np.linalg.multi_dot([self.bvmodel.matrix,
                                                            self.bvmodel.control_mesh])
            

    def fit_template(self, weight_GP, low_smoothing_weight, transmural_weight, rv_thickness):

        # Create valve phantom points
        mitral_points = self.dataset.create_valve_phantom_points(30, ContourType.MITRAL_VALVE)
        tri_points = self.dataset.create_valve_phantom_points(30, ContourType.TRICUSPID_VALVE)
        aorta_points = self.dataset.create_valve_phantom_points(10, ContourType.AORTA_VALVE)

        # Generates RV epicardial point if they have not been contoured
        rv_epi_points,rv_epi_contour, rv_epi_slice = self.dataset.create_rv_epicardium(
            rv_thickness=rv_thickness)

    
        # Plot rigid fit
        contourPlots = self.dataset.PlotDataSet([ContourType.LAX_RA,
                    ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                    ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                    ContourType.SAX_LV_ENDOCARDIAL,
                    ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                    ContourType.APEX_ENDO_POINT, ContourType.APEX_EPI_POINT,
                    ContourType.MITRAL_VALVE, ContourType.TRICUSPID_VALVE,
                    ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                    ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                    ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                    ContourType.PULMONARY_VALVE, ContourType.AORTA_VALVE,
                    ContourType.AORTA_PHANTOM, ContourType.MITRAL_PHANTOM,
                    ContourType.TRICUSPID_PHANTOM,
                    ])
        model = self.bvmodel.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)",
                                    "Initial model", "all")
        
        pu.plot_surface(model, contourPlots, out_path=self.out_prefix + 'step0_fitted.html')

        # Exmple on how to set different weights for different points group
        self.dataset.assign_weights(self.template_fitting_weights)

        # 'Stiff' fit - implicit diffeomorphic constraints
        self.bvmodel.MultiThreadSmoothingED(weight_GP, self.dataset)

        model = self.bvmodel.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","Initial model","all")
        pu.plot_surface(model, contourPlots, out_path=self.out_prefix + 'step1_fitted.html')

        # 'Soft' fit - explicit diffeomorphic constraints
        self.bvmodel.SolveProblemCVXOPT(self.dataset,weight_GP,low_smoothing_weight,transmural_weight)

        model = self.bvmodel.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","Initial model","all")

        pu.plot_surface(model, contourPlots, out_path=self.out_prefix + 'step2_fitted.html')

        self.template_mesh = self.bvmodel.get_template_mesh()


    def get_surface_mesh(self, mesh_subdivisions=2):
        self.bvmesh, self.valve_mesh, self.septum_mesh = self.bvmodel.get_bv_surface_mesh(subdivisions=mesh_subdivisions)


    def save_mesh(self, mesh_subdivisions):
        # Save .stl and control points
        self.get_surface_mesh(mesh_subdivisions=mesh_subdivisions)
        
        io.write(self.out_prefix + 'bv_surface.stl', self.bvmesh)
        io.write(self.out_prefix + 'valve_surfaces.stl', self.valve_mesh)
        io.write(self.out_prefix + 'septum_surface.stl', self.septum_mesh)

        # Save control points
        np.save(self.out_prefix + 'control_points.npy', self.bvmodel.control_mesh)


    def make_volumetric_mesh(self):
        volumetric_ien = np.load(f'{filepath}/template/volume_template_ien.npy')
        bvmesh, _, _ = self.bvmodel.get_bv_surface_mesh()
        mesh = io.Mesh(bvmesh.points, {'tetra': volumetric_ien})
        return mesh
    

class BVSurface:
    def __init__(self, template_mesh):
        self.mesh = template_mesh