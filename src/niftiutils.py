#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:41:24 2023

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np

import monai.transforms as mt
import nibabel as nib


def readFromNIFTI(imgName, return_header=False, return_slice_duration=False):
    ''' Helper function used by masks2ContoursSA() and masks2ContoursLA(). Returns (seg, transform, pixSpacing). '''
    # Load NIFTI image and its header.
    if os.path.isfile(imgName):
        ext = ''
    elif os.path.isfile(imgName + '.nii.gz'):
        ext = '.nii.gz'
    elif os.path.isfile(imgName + '.nii'):
        ext = '.nii'
    else:
        raise FileNotFoundError('File {} was not found'.format(imgName))

    img, header = mt.LoadImage(image_only=False)(imgName + ext)
    data = img.numpy().astype(float)
    transform = img.affine.numpy()
    pixdim = img.pixdim.numpy()

    if return_header:
        return (data, transform, pixdim, header)
    elif return_slice_duration:
        slice_duration = img.meta['slice_duration']
        return (data, transform, pixdim, slice_duration)
    else:
        return (data, transform, pixdim)


def saveInterpNIFTI(ogName, outName, data, slice_duration):
    if os.path.isfile(ogName):
        ext = ''
    elif os.path.isfile(ogName + '.nii.gz'):
        ext = '.nii.gz'
    elif os.path.isfile(ogName + '.nii'):
        ext = '.nii'
    else:
        raise FileNotFoundError('File {} was not found'.format(ogName))
    
    # I don't want to mess up the original image, so it's better to load it with nibabel and save it again
    og = nib.load(ogName + ext)

    header = og.header 
    zooms = list(header.get_zooms())
    zooms[3] = slice_duration
    header.set_zooms(zooms)
    
    # Save the interpolated image
    img = nib.Nifti1Image(data, og.affine, header)
    nib.save(img, outName + '_interp.nii.gz')
