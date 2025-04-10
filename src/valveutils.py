

import numpy as np
from skimage  import morphology
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
    
from niftiutils import readFromNIFTI

def load_valve_nii(filename, view, frame=0, labels = {'mv': 1, 'tv': 2, 'av': 3, 'pv': 4}):
    """
    Load valve points from a NIFTI file based on the specified view.
    Parameters:
    filename (str): Path to the NIFTI file.
    view (str): The view type. Must contain one of these strings: ('2ch', '3ch', or '4ch').
    labels (dict, optional): A dictionary mapping valve names to their corresponding labels in the NIFTI file.
                                Default is {'mv': 1, 'tv': 2, 'av': 3, 'pv': 4}.
    Returns:
    numpy.ndarray or tuple: 
        - If '2ch' view is specified, returns an array of mitral valve points.
        - If '3ch' view is specified, returns a tuple of arrays (mitral valve points, aortic valve points).
        - If '4ch' view is specified, returns a tuple of arrays (mitral valve points, tricuspid valve points).
    Raises:
    AssertionError: If no points are found for the specified valves in the given view.
    """

    try:
        valves = readFromNIFTI(filename)[0][...,frame]
    except FileNotFoundError:
        print(f'File {filename} not found')
        return
    mv_points = np.column_stack(np.where(valves == labels['mv']))
    tv_points = np.column_stack(np.where(valves == labels['tv']))
    av_points = np.column_stack(np.where(valves == labels['av']))
    pv_points = np.column_stack(np.where(valves == labels['pv']))

    if '2ch' in view:
        assert len(mv_points) > 0, 'No mitral valve points found'
        return mv_points
    
    if '3ch' in view:
        if len(mv_points) == 0:
            print('WARNING: No mitral valve points found in 3CH')
        assert len(av_points) > 0, 'No aortic valve points found'
        return mv_points, av_points

    if '4ch' in view:
        assert len(mv_points) > 0, 'No mitral valve points found'
        assert len(tv_points) > 0, 'No tricuspid valve points found'
        return mv_points, tv_points
    

def get_mv_points(seg, slice):
    # Get shape
    nframes = seg.data.shape[-1]

    # Initialize arrays
    mv_points = np.zeros((nframes, 2, 2))
    mv_centroids = np.zeros((nframes, 2))

    for frame in range(nframes):
        data = seg.data[..., slice, frame]

        # Need to find the base. We can find the intersection of the blood pool and the myocardium border.
        lv = data == seg.labels['lv']
        lvbp = data == seg.labels['lvbp']

        lv_lvbp = lv + lvbp
        lv_lvbp_border = np.logical_xor(lv_lvbp, morphology.binary_erosion(lv_lvbp, morphology.disk(1)))
        lvbp_border = np.logical_xor(lvbp, morphology.binary_erosion(lvbp, morphology.disk(1)))
        mv_border = lvbp_border*lv_lvbp_border
        
        # Need to find the edge points
        mv_border_points = np.column_stack(np.where(mv_border))
        mv_centroid = np.mean(np.where(mv_border), axis=1)
        
        mv_points_dist = np.linalg.norm(mv_border_points - mv_centroid, axis=1)
        mv_edge1 = mv_border_points[np.argmax(mv_points_dist)]
        vector = mv_edge1 - mv_centroid
        vector = - vector / np.linalg.norm(vector)

        mv_points_dist = vector @ (mv_border_points - mv_centroid).T
        mv_edge2 = mv_border_points[np.argmax(mv_points_dist)]

        # If 3CH or 4CH check which point is lateral
        if seg.view in ['la_3ch', 'la_4ch']:
            rv_cent = np.mean(np.column_stack(np.where(data == seg.labels['rv'])), axis=0)
            if np.linalg.norm(mv_edge1 - rv_cent) < np.linalg.norm(mv_edge2 - rv_cent):
                mv_points[frame] = np.vstack([mv_edge2, mv_edge1])
            else:
                mv_points[frame] = np.vstack([mv_edge1, mv_edge2])
        else:
            mv_points[frame] = np.vstack([mv_edge1, mv_edge2])
        mv_centroids[frame] = np.mean(mv_points[frame], axis=0)

    return mv_points, mv_centroids


def get_tv_points(seg, tv_seg_points, slice):
    # Get shape
    nframes = seg.data.shape[-1]

    # Initialize arrays
    tv_points = np.zeros((nframes, 2, 2))
    tv_centroids = np.zeros((nframes, 2))

    # Find points by looking at the closer point in the next frame
    tv_points_fwd = np.zeros((nframes, 2, 2))
    tv_centroids_fwd = np.zeros((nframes, 2))
    for frame in range(nframes):
        data = seg.data[..., slice, frame]
        rv = data == seg.labels['rv']
        lv = data == seg.labels['lv']
        
        # Get rv border
        rv_border = rv ^ morphology.binary_erosion(rv)
        rv_border[morphology.binary_dilation(lv)] = 0

        rv_points = np.column_stack(np.where(rv_border))

        tree = KDTree(rv_points)
        if frame == 0:
            _, ind = tree.query(tv_seg_points)
        else:
            _, ind = tree.query(tv_points_fwd[frame - 1])

        tv_points_fwd[frame] = rv_points[ind]
        tv_centroids_fwd[frame] = np.mean(tv_points_fwd[frame], axis=0)


    # Same but backwards
    tv_points_bwd = np.zeros((nframes, 2, 2))
    tv_centroids_bwd = np.zeros((nframes, 2))

    for frame in reversed(range(nframes)):
        data = seg.data[..., slice, frame]
        rv = data == seg.labels['rv']
        lv = data == seg.labels['lv']
        
        # Get rv border
        rv_border = rv ^ morphology.binary_erosion(rv)
        rv_border[morphology.binary_dilation(lv)] = 0

        rv_points = np.column_stack(np.where(rv_border))

        tree = KDTree(rv_points)
        if frame == nframes-1:
            _, ind = tree.query(tv_seg_points)
        else:
            _, ind = tree.query(tv_points_bwd[frame+1])

        tv_points_bwd[frame] = rv_points[ind]
        tv_centroids_bwd[frame] = np.mean(tv_points_bwd[frame], axis=0)

    # Combine fwd and backward
    tv_centroids_disp_fwd = tv_centroids_fwd - tv_centroids_fwd[0]
    tv_centroids_disp_bwd = tv_centroids_bwd - tv_centroids_bwd[-1]
    mag_disp_fwd = np.linalg.norm(tv_centroids_disp_fwd, axis=1)
    mag_disp_bwd = np.linalg.norm(tv_centroids_disp_bwd, axis=1)

    peak_fwd = np.argmax(mag_disp_fwd)

    tv_points = np.copy(tv_points_fwd)
    tv_points[peak_fwd:] = tv_points_bwd[peak_fwd:]

    tv_centroids = np.mean(tv_points, axis=1)
    
    return tv_points, tv_centroids


def get_3ch_valve_points(seg, mv_seg_points, av_seg_points, slice):
    # Get shape
    nframes = seg.data.shape[-1]

    # Find bridge points
    dist = cdist(mv_seg_points, av_seg_points)
    mv_bridge_ind, av_bridge_ind = np.unravel_index(np.argmin(dist), dist.shape)
    mv_lat_ind = 1 - mv_bridge_ind
    av_lat_ind = 1 - av_bridge_ind
    bridge_seg_point = (mv_seg_points[mv_bridge_ind] + av_seg_points[av_bridge_ind]) / 2

    mv_bridge_vector = mv_seg_points[mv_bridge_ind] - bridge_seg_point
    av_bridge_vector = av_seg_points[av_bridge_ind] - bridge_seg_point

    # First find lvbp and lv intersection points
    inter_points_, _ = get_mv_points(seg, slice=slice)
    inter_points = inter_points_  # squeezing slice dim

    av_centroid = np.mean(av_seg_points, axis=0)

    av_inter_disp = np.zeros((nframes, 2))
    mv_inter_disp = np.zeros((nframes, 2))

    av_inter_ind = 0
    mv_inter_ind = 1
    for frame in range(nframes):
        # Find which points are closer to each valve
        inter_points_0 = inter_points[frame]

        if frame == 0:
            av_dist = np.linalg.norm(av_centroid - inter_points_0, axis=1)
        else:
            av_dist = np.linalg.norm(inter_points[frame-1, av_inter_ind] - inter_points_0, axis=1)

        av_inter_ind = np.argmin(av_dist)
        mv_inter_ind = 1 - av_inter_ind
        
        if frame == 0:
            mv_inter_ind0 = mv_inter_ind
            av_inter_ind0 = av_inter_ind

        av_inter_disp[frame] = inter_points[frame, av_inter_ind] - inter_points[0, av_inter_ind0]
        mv_inter_disp[frame] = inter_points[frame, mv_inter_ind] - inter_points[0, mv_inter_ind0]

    # Calculate bridge and lateral points displacement
    av_lat_point = np.zeros((nframes, 2))
    mv_lat_point = np.zeros((nframes, 2))
    bridge_point = np.zeros((nframes, 2))

    av_lat_point[0] = av_seg_points[av_lat_ind]
    mv_lat_point[0] = mv_seg_points[mv_lat_ind]
    bridge_point[0] = bridge_seg_point

    for i in range(1, nframes):
        av_lat_point[i] = av_lat_point[0] + av_inter_disp[i]
        mv_lat_point[i] = mv_lat_point[0] + mv_inter_disp[i]
        bridge_point[i] = bridge_point[0] + (av_inter_disp[i] + mv_inter_disp[i]) / 2

    # Calculate position of mv and av at the bridge
    av_bridge_point = bridge_point + av_bridge_vector
    mv_bridge_point = bridge_point + mv_bridge_vector

    # Save points
    av_points = np.zeros((nframes, 2, 2))
    mv_points = np.zeros((nframes, 2, 2))

    av_points[:,0] = av_lat_point
    av_points[:,1] = av_bridge_point

    mv_points[:,0] = mv_lat_point
    mv_points[:,1] = mv_bridge_point

    # Calculate centroids
    mv_centroids = np.mean(mv_points, axis=1)
    av_centroids = np.mean(av_points, axis=1)

    return mv_points, mv_centroids, av_points, av_centroids


def plot_valve_movement(img, seg, slice=0, valve_points={}, valve_centroids={}):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initial plot
    frame = 0
    img_display = ax.imshow(img.data[..., slice, frame], cmap='gray')
    seg_display = ax.imshow(np.ma.masked_where(seg.data[..., slice, frame] == 0, seg.data[..., slice, frame]), cmap='jet', alpha=0.5)

    # Plot valve points and centroids
    color = ['mo', 'co']
    point_displays = {}
    for key, item in valve_points.items():
        for i in range(item[slice].shape[1]):
            point_displays[key] = ax.plot(item[slice][frame, i, 1], item[slice][frame, i, 0], color[i], label=key)[0]

    centroid_displays = {}
    for key, item in valve_centroids.items():
        centroid_displays[key] = ax.plot(item[slice][frame, 1], item[slice][frame, 0], 'ro', label=key)[0]
    
    # Slider axis
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, seg.data.shape[-1] - 1, valinit=0, valstep=1)

    # Update function
    def update(val):
            frame = int(slider.val)
            img_display.set_data(img.data[..., slice, frame])
            seg_display.set_data(np.ma.masked_where(seg.data[..., slice, frame] == 0, seg.data[..., slice, frame]))

            for key, item in valve_points.items():
                for i in range(item[slice].shape[1]):
                    point_displays[key] = ax.plot(item[slice][frame, i, 1], item[slice][frame, i, 0], color[i], label=key)[0]

            for key, item in valve_centroids.items():
                centroid_displays[key].set_data([item[slice][frame, 1]], [item[slice][frame, 0]])
                
            fig.canvas.draw_idle()

    # Attach update function to slider
    slider.on_changed(update)

    plt.show()

