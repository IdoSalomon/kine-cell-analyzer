import os

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


def create_stack(chan_paths):
    channels = {}
    # Scan all channels
    for chan_path in chan_paths:
        # Check if channel is monitored
        for chan_type in channel_types:
            # Add relevant image to channel
            if chan_type in chan_path:
                channels[chan_type] = img_utils.load_img(chan_path, 0.5, False, False)
                break

    return channels


def create_masks(channels, ker_params, opt_params, interval):
    chans = {}
    chans["PHASE"] = pr.seg_phase(channels["PHASE"], ker_params=ker_params, opt_params=opt_params, file_name=interval, debug=True)
    return channels


def get_cells_con_comps(con_comps, debug=True):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=connectivity)

    #TODO
    pass


def load_frame(interval, ker_params, opt_params, seq_paths, debug=True):
    images = create_stack(seq_paths[interval])
    masks = create_masks(images, ker_params=ker_params, opt_params=opt_params, interval=interval)
    con_comps = pr.getConnectedComponents(masks, grayscale=True, debug=debug)
    cells = get_cells_con_comps(con_comps, debug=True)
    frame = fr.Frame(num=io_utils.extract_num(interval), title=interval, images=images, masks=masks, cells=cells, con_comps=con_comps)
    print("Loaded frame {}\n".format(interval))

    return frame


def load_sequence(dir, ker_params, opt_params):
    seq_paths = io_utils.load_paths(dir)

    for interval in seq_paths:
        frame = load_frame(interval, ker_params=ker_params, opt_params=opt_params, seq_paths=seq_paths)

        seq_frames[frame.num] = frame

    print("Finished loading sequence!\n") # DEBUG

def save_con_comps():
    print("todo")
    # TODO

def load_tracked_masks(): # TODO remove
    print("load_tracked_masks\n")
    #for frame in seq_frames:
        #frame.tracked_mask = img_utils.load_img(tracked_mask_path, 0.5, False, False)

        #seq_frames[frame.num] = frame

if __name__ == "__main__":
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)

    load_sequence("images\\seq_nec", ker_params=ker_params, opt_params=opt_params)
    #save_con_comps()
