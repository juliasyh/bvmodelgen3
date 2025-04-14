#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/04/12 15:34:20

@author: Javiera Jilberto Vallejos 
'''

import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from niftiutils import readFromNIFTI, saveInterpNIFTI

def register_image(moving, target):
    """
    Register a moving image to a target image using the Demons registration algorithm.

    Parameters:
    moving (np.ndarray): The moving image as a numpy array to be registered.
    target (np.ndarray): The target image as a numpy array to which the moving image will be registered.

    Returns:
    np.ndarray: The displacement field as a numpy array representing the transformation needed to align the moving image to the target image.
    """

    # Convert numpy arrays to SimpleITK images
    moving_sitk = sitk.GetImageFromArray(moving)
    target_sitk = sitk.GetImageFromArray(target)

    # Initialize the Demons registration filter
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(50)
    demons.SetStandardDeviations(1.0)

    # Perform the registration
    displacement_field = demons.Execute(moving_sitk, target_sitk)

    # Get transform
    transform = sitk.DisplacementFieldTransform(displacement_field)

    # Convert the displacement field back to a numpy array
    displacement_field = sitk.GetArrayFromImage(transform.GetDisplacementField())

    return displacement_field


def warp_image(moving, displacement_field):
    """
    Warp a moving image using a displacement field.

    Parameters:
    moving (np.ndarray): The moving image as a numpy array to be warped.
    displacement_field (np.ndarray): The displacement field as a numpy array representing the transformation.

    Returns:
    np.ndarray: The warped image as a numpy array.
    """

    # Convert numpy array to SimpleITK image
    moving_sitk = sitk.GetImageFromArray(moving)

    # Get transform
    transform = sitk.DisplacementFieldTransform(sitk.GetImageFromArray(displacement_field, isVector=True))

    # Warp the moving image using the displacement field
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # Execute the warping
    warped_image_sitk = resampler.Execute(moving_sitk)

    # Convert the warped image back to a numpy array
    warped_image = sitk.GetArrayFromImage(warped_image_sitk)

    return warped_image

def interpolate_frames(img_dn, img_up, alpha):
    """
    Interpolate between two images using a linear combination.

    Parameters:
    img_dn (np.ndarray): The first image as a numpy array.
    img_up (np.ndarray): The second image as a numpy array.
    alpha (float): The interpolation factor (0 <= alpha <= 1).

    Returns:
    np.ndarray: The interpolated image as a numpy array.
    """

    # Get the displacement fields
    disp_up = register_image(img_dn, img_up)
    disp_dn = register_image(img_up, img_dn)

    # Warp the images using the displacement fields
    warped_up = warp_image(img_dn, disp_up * alpha)
    warped_dn = warp_image(img_up, disp_dn * (1 - alpha))

    # Interpolate between the two images
    warped_img = warped_up * (1-alpha) + warped_dn * alpha

    return warped_img


def interpolate_img(img, nframes):
    """
    Interpolate a 4D image to a specified number of frames.

    Parameters:
    img (np.ndarray): The original 4D image as a numpy array.
    nframes (int): The desired number of frames for interpolation.

    Returns:
    np.ndarray: The interpolated 4D image as a numpy array.
    """

    nframes_og = img.shape[3]
    timepoints_og = np.linspace(0, 1, nframes_og)
    timepoints = np.linspace(0, 1, nframes)

    interp_img = np.zeros((img.shape[0], img.shape[1], img.shape[2], nframes))
    i_og = 0
    for i, time in tqdm(enumerate(timepoints)):
        # Check if we are matching the original timepoints
        if np.isclose(time, timepoints_og[i_og]):
            # Use the original image
            interp_img[:, :, :, i] = img[:, :, :, i_og]
            i_og += 1
            continue

        if time > timepoints_og[i_og]:
            i_og += 1

        alpha = (time - timepoints_og[i_og-1]) / (timepoints_og[i_og] - timepoints_og[i_og-1])

        for slice in range(img.shape[2]):
            # Get the two images to interpolate between
            img_dn = img[:, :, slice, i_og-1]
            img_up = img[:, :, slice, i_og]

            # Interpolate between the two images
            interp_img[:, :, slice, i] = interpolate_frames(img_dn, img_up, alpha)

    return interp_img


def interpolate_scans(img_paths, nframes):
    """
    Interpolates a set of 4D NIFTI images to a specified number of frames.
    Parameters:
    -----------
    img_paths : dict
        A dictionary where keys are view identifiers (e.g., 'sa', 'la_2ch', etc.) 
        and values are the file path to the corresponding NIFTI image.
    nframes : int or str
        The target number of frames for interpolation. Can be:
        - An integer specifying the exact number of frames.
        - 'max' to interpolate to the maximum number of frames among the input images.
        - 'min' to interpolate to the minimum number of frames among the input images.
    Returns:
    --------
    list
        A list of views (keys from `img_paths`) for which interpolation was performed. 
        If no interpolation was needed, an empty list is returned.
    Raises:
    -------
    ValueError
        If `nframes` is not an integer, 'max', or 'min'.
    Notes:
    ------
    - The function reads 4D NIFTI images, determines their slice durations, and 
      interpolates them to the desired number of frames.
    - If all input images already have the same number of frames as `nframes`, 
      no interpolation is performed.
    - Interpolated images are saved with a modified filename, appending '_interp' 
      before the file extension.
    """

    # Read the images and their slice durations
    imgs = {}
    imgs_sdur = {}
    img_frames = []
    print(img_paths)
    for view, img_path in img_paths.items():
        img, _, _, header = readFromNIFTI(img_path, return_header=True)
        imgs[view] = img
        imgs_sdur[view] = header['slice_duration']
        img_frames.append(img.shape[3])

    # Check nframes options
    if nframes == 'max':
        nframes = max(img_frames)
        print(f'Interpolating to {nframes} frames.')
    elif nframes == 'min':
        nframes = min(img_frames)
        print(f'Interpolating to {nframes} frames.')
    elif isinstance(nframes, int):
        print(f'Interpolating to {nframes} frames.')
    else:
        raise ValueError('nframes must be an int, "max" or "min".')
    
    # Check if all images have the same number of frames
    if len(np.unique(img_frames)) == 1:
        if img_frames[0] == nframes:
            print('All images already have the required number of frames. No interpolation needed.')
            return []

    # Interpolate the images
    interpolated_images = []
    for view, img_path in img_paths.items():
        img, _, _, header = readFromNIFTI(img_path, return_header=True)
        nframes_og = img.shape[3]
        if nframes_og == nframes:
            print(f'Image {img_path} already has {nframes} frames. No interpolation needed.')
            continue

        interp_img = interpolate_img(img, nframes)

        # Get total cycle time
        tcycle = header['slice_duration']*nframes_og

        # Save the interpolated image
        path = os.path.dirname(img_path)
        fname = os.path.basename(img_path)
        if 'nii.gz' in fname:
            fname = fname.replace('.nii.gz', '')
        elif 'nii' in fname:
            fname = fname.replace('.nii', '')

        saveInterpNIFTI(img_path, f'{path}/{fname}', interp_img, tcycle/nframes)

        interpolated_images.append(view)

    return interpolated_images