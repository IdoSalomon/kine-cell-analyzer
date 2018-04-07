import os

import img_utils
import img_utils as iu
import numpy as np
import prec_sparse as ps
import process as pr
import frame as fr
import io_utils

#seq_paths = {} # Paths to all sequence images
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


def create_masks(channels, interval):
    pr.seg_phase(channels["TRANS"], file_name=interval, debug=True)
    # TODO
    return channels


def load_frame(interval, seq_paths):
    images = create_stack(seq_paths[interval])
    masks = create_masks(images, interval)

    frame = fr.Frame(num=io_utils.extract_num(interval), title=interval, images=images)
    print("Loaded frame {}\n".format(interval))

    return frame


def load_sequence(dir):
    seq_paths = io_utils.load_paths(dir)

    for interval in seq_paths:
        frame = load_frame(interval, seq_paths)

        seq_frames[frame.num] = frame

    print("Finished loading sequence!\n") # DEBUG



if __name__ == "__main__":
    load_sequence("images\\seq_apo")
