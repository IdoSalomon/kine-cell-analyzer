import os

import cv2

import cell
import img_utils
import img_utils as iu
import numpy as np
import prec_sparse as ps
import process as pr
import frame as fr
import io_utils

#seq_paths = {} # Paths to all sequence images
from prec_params import KerParams, OptParams

seq_frames = {} # dictionary <str,Frame> that holds all frames in sequence by number

channel_types = ["GFP", "PHASE", "TxRed", "TRANS"] # different channels in sequence TODO turn to user input

"""def load_paths(dir):
    global seq_paths = io_utils.load_paths(dir)

    print("Loaded paths!\n") # DEBUG"""


def create_stack(chan_paths, opt_params):
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


def load_tracked_mask(tracked_path, opt_params, grayscale=True,debug=False):
    """tracked_mask = img_utils.load_img(tracked_path["PHASE"], opt_params.img_scale, False, False)
    con_comps = pr.get_connected_components(tracked_mask, grayscale=True, debug=debug)

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
        cells[frame_label] = cell.Cell(frame_label=frame_label, area=area, centroid=centroid, pixels=pixels)"""


def create_masks(channels, ker_params, opt_params, interval):
    chans = {}
    chans["PHASE"] = pr.seg_phase(channels["PHASE"], despeckle_size=6, filter_size=0, ker_params=ker_params, opt_params=opt_params, file_name=interval, debug=True)
    return chans


def get_cells_con_comps(con_comps, debug=True):
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


def load_frame(interval, tracked_paths, ker_params, opt_params, seq_paths, debug=True):
    images = create_stack(seq_paths[interval], opt_params=opt_params)
    masks = create_masks(images, ker_params=ker_params, opt_params=opt_params, interval=interval)
    con_comps = pr.get_connected_components(masks["PHASE"], grayscale=True, debug=debug)
    cells = get_cells_con_comps(con_comps, debug=True)

    io_utils.save_img(con_comps[1], "images\\seq_nec\\concomps\\" + interval + ".png") # TODO Remove
    #tracked_mask = load_tracked_mask(seq_paths[interval], cells) # TODO Remove

    frame = fr.Frame(num=io_utils.extract_num(interval), title=interval, images=images, masks=masks, cells=cells, con_comps=con_comps)
    print("Loaded frame {}\n".format(interval))

    return frame


def load_sequence(dir, ker_params, opt_params, dir_mask):
    seq_paths = io_utils.load_paths(dir)
    tracked_paths = io_utils.load_paths(dir_mask) # TODO Remove

    for interval in seq_paths:
        frame = load_frame(interval, ker_params=ker_params, opt_params=opt_params, seq_paths=seq_paths, tracked_paths=tracked_paths)

        seq_frames[frame.num] = frame

    print("Finished loading sequence!\n") # DEBUG

def save_con_comps(dir):
    print("todo")
    # TODO

def load_tracked_masks(dir): # TODO remove
    print("load_tracked_masks\n")
    tracked_file_paths = io_utils.load_paths(dir)


    #for frame in seq_frames:
        #frame.tracked_mask = img_utils.load_img(tracked_mask_path, 0.5, False, False)

        #seq_frames[frame.num] = frame

if __name__ == "__main__":
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)

    load_sequence("images\\seq_nec", ker_params=ker_params, opt_params=opt_params, dir_mask="images\\seq_nec\\tracked")


#save_con_comps()
