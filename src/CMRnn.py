#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/11/16 15:58:47

@author: Javiera Jilberto Vallejos 
'''
import os
import glob

import numpy as np
import nibabel as nib

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

dataset_by_view = {'sa': 'Dataset041_ACDC',
                   'la_2ch': 'Dataset042_2CH',
                   'la_3ch': 'Dataset043_LA_3CH',
                   'la_4ch': 'Dataset044_LA_4CH'}

configuration_by_view = {'sa': '3d_fullres',
                            'la_2ch': '2d',
                            'la_3ch': '2d',
                            'la_4ch': '2d'}

def split_nifti(filepath):
    img = nib.load(filepath)

    fldrpath = os.path.dirname(filepath)
    
    if 'nii.gz' in filepath:
        filename = os.path.basename(filepath)[:-7]
    else:
        filename = os.path.basename(filepath)[:-4]

    # Split nifti file
    img = nib.load(filepath)
    data = img.get_fdata()
    assert len(data.shape) == 4

    if not os.path.exists(f'{fldrpath}/imgs/'):
        os.mkdir(f'{fldrpath}/imgs/')

    for fr in range(data.shape[-1]):
        new_img = nib.Nifti1Image(data[:,:,:,fr], img.affine, header=img.header)
        nib.save(new_img, f'{fldrpath}/imgs/{filename}_{fr:03d}_0000.nii.gz')

    return f'{fldrpath}/imgs/'


def predict_nifti(input_fldr, view):
    output_fldr = f'{input_fldr}/../segs/'
    if not os.path.exists(output_fldr):
        os.mkdir(output_fldr)

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, f'{dataset_by_view[view]}/nnUNetTrainer__nnUNetPlans__{configuration_by_view[view]}'),
        use_folds=(0,1,2,3,4),
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(input_fldr,
                                 output_fldr,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
    return output_fldr

def join_nifti(fldr, view):
    files = glob.glob(f'{fldr}/*.nii.gz')
    files = list(np.sort(files))

    nframes = len(files)

    # Read first to get data size
    img = nib.load(files[0])
    data_fr = img.get_fdata()

    data = np.zeros([*data_fr.shape, nframes], dtype=int)
    data[:,:,:,0] = data_fr

    for i, file in enumerate(files[1:]):
        fr = i+1
        img = nib.load(file)
        data_fr = img.get_fdata()
        data[:,:,:,fr] = data_fr

    new_img = nib.Nifti1Image(data, img.affine, header=img.header)
    nib.save(new_img, f'{fldr}/../{view}_seg.nii.gz')


def clean_fldr(working_fldr):
    os.system(f'rm -r {working_fldr}/segs/')
    os.system(f'rm -r {working_fldr}/imgs/')