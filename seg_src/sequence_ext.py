
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
import cell as cl

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

    if tracked_path and opt_params:
        return img_utils.load_img(tracked_path[0], 1, False, False, float=False, normalize=False)


def get_cells_ext(tracked_img, images, frame_id):
    """
    Loads cells from tracked image
    Parameters
    ----------
    tracked_img : ndarray
        2D array of tracked image
    images : dict<str, ndarray>
        dictionary that holds channel images by channel name
    frame_id : str
        ID of frame
    Returns
    -------
    cells : dict<int, Cell>
        dictionary of cells by cell ID
    """

    cells = {}
    channels_pixels = {}

    # Find cells in tracked image
    labels = np.unique(tracked_img)

    # Create cell for each label in tracked image
    for label in labels:
        # Find pixel values for each channel
        for channel in images:
            channels_pixels[channel] = (images[channel])[tracked_img == label]
        cell = cl.Cell(global_label=label, frame_label=label, pixel_values=channels_pixels)
        cells[label] = cell

        # Assign frame to gloal label
        if label not in cells_frames:
            cells_frames[label] = [frame_id]
        else:
            cells_frames[label].append(frame_id)

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

    frame_id = io_utils.extract_num(interval)
    images = create_stack(seq_paths[interval], opt_params=opt_params) # original channels
    tracked_img = load_tracked_mask(tracked_paths["trk-" + interval], opt_params) # image after tracking
    cells = get_cells_ext(tracked_img=tracked_img, images=images, frame_id=frame_id) # image's cell representation

    frame = fr.Frame(id=frame_id, title=interval, images=images, cells=cells, tracked_img=tracked_img)
    print("Loaded frame {}\n".format(interval))

    return frame


def load_sequence_ext(dir, opt_params, dir_tracked):
    """
    Loads sequence of frames.

    Parameters
    ----------
    dir : str
        path to image directory
    opt_params : OptParams
        optimization parameters
    dir_tracked : str
        path to tracked images directory

    """
    seq_paths = io_utils.load_paths(dir)
    tracked_paths = io_utils.load_paths(dir_tracked)
    for interval in seq_paths:
        frame = load_frame_ext(interval, opt_params=opt_params, seq_paths=seq_paths, tracked_paths=tracked_paths)

        seq_frames[frame.id] = frame

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
    load_sequence_ext("images\\seq_nec", opt_params=opt_params, dir_tracked="images\\seq_nec\\concomps\\track")


#save_con_comps()
