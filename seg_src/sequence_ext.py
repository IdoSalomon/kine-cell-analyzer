
import os
import random

import cv2

import cell
import img_utils
import img_utils as iu
import numpy as np
import prec_sparse as ps
import process as pr
import frame as fr
import io_utils

from prec_params import KerParams, OptParams

seq_frames = {} # dictionary <int,Frame> that holds all frames in sequence by number

channel_types = ["GFP", "PHASE", "TxRed", "TRANS"] # different channels in sequence TODO turn to user input

cells_frames = {} # dictionary <int, list<int>> that holds IDs of frame in which the cell appeared

cells_trans = {} # dictionary <int, dict<str, int>> that holds dictionary of interval IDs by channels for each cell's transformations

def create_stack(chan_paths, opt_params):
    """
    Loads images for all channels.

    Parameters
    ----------
    chan_paths : list
        paths to channel images
    opt_params : OptParams
        optimization parameters
    Returns
    -------
        channels : dictionary<str, ndarray>
            dictionary that holds images by channel name
    """
    channels = {}
    # Scan all channels
    for chan_path in chan_paths:
        # Check if channel is monitored
        for chan_type in channel_types:
            # Add relevant image to channel
            if chan_type in chan_path:
                channels[chan_type] = img_utils.load_img(chan_path, opt_params.img_scale, False, False)
                break

    return channels


def load_tracked_mask(tracked_path, opt_params):
    """
    Loads tracked image.

    Parameters
    ----------
    tracked_path : str
        path to tracked image file
    opt_params : OptParams
        optimization parameters

    Returns
    -------
        img : ndarray
            2D array of tracked image
    """


def get_cells_ext(tracked_img):
    """
    Loads cells from tracked image
    Parameters
    ----------
    tracked_img : ndarray
        2D array of tracked image

    Returns
    -------
    cells : list
        list of cells
    """
    cells = {}
    num_labels = con_comps[0]
    label_mat = con_comps[1]
    stats = con_comps[2]
    centroids = con_comps[3]

    for i in range(1, num_labels):
        frame_label = i
        area = stats[i][4]
        centroid = round(centroids[i][0]), round(centroids[i][1])
        pixels = np.where(label_mat == i)
        pixels = list(zip(pixels[0], pixels[1]))
        cells[frame_label] = cell.Cell(frame_label=frame_label, area=area, centroid=centroid, pixels=pixels)
    return cells


def load_frame_ext(interval, tracked_paths, opt_params, seq_paths):
    """
    Loads frame.

    Parameters
    ----------
    interval : str
        interval name
    tracked_paths : list
        paths to tracked images
    opt_params : OptParams
        optimization parameters
    seq_paths : list
        paths to sequence images

    Returns
    -------
    frame : Frame
        Loaded frame
    """
    images = create_stack(seq_paths[interval], opt_params=opt_params)
    cells = get_cells_con_comps(con_comps, debug=True)

    #tracked_mask = load_tracked_mask(seq_paths[interval], cells) # TODO Remove

    frame = fr.Frame(num=io_utils.extract_num(interval), title=interval, images=images, masks=masks, cells=cells, con_comps=con_comps)
    print("Loaded frame {}\n".format(interval))

    return frame


def load_sequence_ext(dir, ker_params, opt_params, dir_tracked):
    """
    Loads sequence of frames.

    Parameters
    ----------
    dir : str
        path to image directory
    ker_params : KerParams
        kernel parameters
    opt_params : OptParams
        optimization parameters
    dir_tracked : str
        path to tracked images directory

    """
    seq_paths = io_utils.load_paths(dir)
    tracked_paths = io_utils.load_paths(dir_tracked) # TODO Remove
    for interval in seq_paths:
        frame = load_frame_ext(interval, ker_params=ker_params, opt_params=opt_params, seq_paths=seq_paths, tracked_paths=tracked_paths, debug = False)

        seq_frames[frame.num] = frame

    print("Finished loading sequence!\n") # DEBUG


def debug_channel(img, colors, name):
    """

    Parameters
    ----------
    img
    colors
    name

    Returns
    -------

    """
    labeled_img = np.zeros_like(img)

    # Map component labels to hue val
    for i in range(1, np.max(img)):
        labeled_img[img == i] = label_colors[i]
    blank_ch = 255 * np.ones_like(labeled_img)
    labeled_img = cv2.merge((labeled_img, blank_ch, blank_ch))

    labeled_img = np.uint8(labeled_img)
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[img == 0] = 0

    io_utils.save_img(labeled_img, "images\\seq_nec\\concomps\\col_track\\" + name)  # TODO Remove

    # labeled_img = np.zeros_like(labeled_img)
    #
    # # Map component labels to hue val
    # for i in range(1, int((np.max(labeled_img)))):
    #     labeled_img[labeled_img == i] = label_colors[i] * (labeled_img[labeled_img == i] / labeled_img[labeled_img == i][0])
    # blank_ch = 255 * np.ones_like(labeled_img)
    # labeled_img = cv2.merge((labeled_img, blank_ch, blank_ch))
    #
    # labeled_img = np.uint8(labeled_img)
    # # cvt to BGR for display
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # # set bg label to black
    # labeled_img[labeled_img == 0] = 0
    #
    # io_utils.save_img(labeled_img, "images\\seq_nec\\concomps\\col_track\\" + name + ".png")  # TODO Remove
    # #cv2.imshow('labeled.png', labeled_img)
    # #cv2.waitKey()


if __name__ == "__main__":
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                            max_itr=100, opt_tolr=np.finfo(float).eps)

    # load_tracked_masks("images\\seq_nec\\tracked")
    #
    # load_sequence("images\\seq_nec", ker_params=ker_params, opt_params=opt_params, dir_mask="images\\seq_nec\\tracked")
    load_tracked_masks("images\\seq_nec\\concomps\\track", opt_params)


#save_con_comps()
