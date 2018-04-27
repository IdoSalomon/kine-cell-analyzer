
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

cells_trans = {} # dictionary <int, dict<str, int>> that holds for each cell ID the transformation frame ID by channel

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
                img = img_utils.load_img(chan_path, opt_params.img_scale, float=False, normalize=False)
                channels[chan_type] = seg_aux_channels(img)
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

    # Find cells in tracked image
    labels = np.unique(tracked_img)

    # Create cell for each label in tracked image
    for label in labels:
        channels_pixels = {}
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

def seg_aux_channels(img):
    img = iu.im2double(img)
    img = iu.bg_removal(img)

    # normalize after background removal for threshold
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = np.uint8(img)

    # perform OTSU thresholding
    tmp, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img



def analyze_channels(channels):
    """

    For each cell appeared in the 1st frame, finds the first frameId in which
    the cell is colored, repeated for each channel, and updates the cell_trans dictionary.

    Parameters
    ----------
    channels : str
        a string representation of the analyzed channeled, e.g 'GFP'
    """
    for channel in channels: # FIXME This kills performance
        frame_bg = {}  # dictionary <img, float, float> that holds the current frame channel + mean background

        # iterate over first frame identified cells
        for cell in seq_frames[1].cells:
            if cell != 0: # skip background label
                # iterate over next frames
                for frame_id in range(2, max(seq_frames)):
                    if frame_id in cells_frames[cell]:
                        label = seq_frames[frame_id].cells[cell].global_label
                        # if cell is color has changed - update db
                        if check_changed(frame_id, frame_bg, label, channel):
                            if cell not in cells_trans:
                                cells_trans[cell] = {}
                            cells_trans[cell][channel] = frame_id
                            break


def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)


def check_changed(frame_id, frame_stat, label, channel):

    # calculate cell average intensity, if it is substantially larger than background -> decide cell is colored
    cell = seq_frames[frame_id].cells[label]
    cell_mean = np.mean(cell.pixel_values[channel])
    if cell_mean > 25: # 10% covered
        return True
    else:
        return False

def debug_channels(dir, channels):
    """

    Parameters
    ----------
    img
    colors
    name

    Returns
    -------

    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    for channel in channels:
        for frame_id in range(1, max(seq_frames)):
            frame_chan = seq_frames[frame_id].images[channel]
            dbg_frame = np.zeros_like(frame_chan)
            for cell in seq_frames[frame_id].cells:
                if cell in cells_trans and channel in cells_trans[cell]:
                    if cells_trans[cell][channel] <= frame_id:
                        cell_mask = seq_frames[frame_id].tracked_img == cell
                        dbg_frame[cell_mask] = 255
            path = dir + "\\" + channel + str(frame_id) + ".png"
            thresh_path = dir + "\\" + channel + str(frame_id) + "_THRESH.png"
            io_utils.save_img(dbg_frame, path)
            io_utils.save_img(seq_frames[frame_id].images[channel], thresh_path, float=True)



if __name__ == "__main__":
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                            max_itr=100, opt_tolr=np.finfo(float).eps)

    # load_tracked_masks("images\\seq_nec\\tracked")
    #
    load_sequence_ext("images\\seq_nec", opt_params=opt_params, dir_tracked="images\\seq_nec\\concomps\\track")

    analyze_channels(["TxRed", "GFP"])

    print ("Finished analyze_channels\n")

    debug_channels("dbg\\chan_analysis", ["TxRed", "GFP"])

    print ("Finished debug_channels\n")

#save_con_comps()
