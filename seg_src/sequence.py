import os
import random

import cv2

import cell
import img_utils
import img_utils as iu
import misc_params
import sequence_ext as ext
import numpy as np
import prec_sparse as ps
import process as pr
import frame as fr
import io_utils
import subprocess
import cell as cl
import misc_params as mpar
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp

seq_paths = {} # Paths to all sequence images
from prec_params import KerParams, OptParams
from typing import Tuple
from multiprocessing import Pool

seq_paths = {}  # Paths to all sequence images

seq_frames = {}  # dictionary <int,Frame> that holds all frames in sequence by number

label_colors = {}  # dictionary <int, int> that holds constant colors for each labelled cells

channel_types = ["GFP", "FITC", "fitc", "pi", "PI", "PHASE", "phase", "TxRed", "TRANS",
                 "trans"]  # different channels in sequence TODO turn to user input

seg_channel_types = ["PHASE", "TRANS", "phase"]  # different channels in sequence to segment TODO turn to user input

aux_channel_types = ["GFP", "TxRed", "fitc", "FITC", "PI", "pi"]

cells_frames = {}  # dictionary <int, list<int>> that holds IDs of frame in which the cell appeared

cells_trans = {}  # dictionary <int, dict<str, int>> that holds for each cell ID the transformation frame ID by channel

diff = {}


def create_stack(chan_paths, opt_params):
    """
    Loads all frame's channel images.
    Parameters
    ----------
    chan_paths : list<str>
        Paths to frame channel images
    opt_params : OptParams
        Optimization parameters

    Returns
    -------
    channels : dict<str, 2darray>
        Dictionary of images by channel name

    """
    channels = {}
    # Scan all channels
    for chan_path in chan_paths:
        # Check if channel is monitored
        for chan_type in channel_types:
            # Add relevant image to channel
            if chan_type in chan_path:
                channels[chan_type] = img_utils.load_img(chan_path, opt_params.img_scale, False, False)
                if chan_type in aux_channel_types:
                    ext.seg_aux_channels(channels[chan_type], chan_type)
                break

    return channels


def create_masks(channels, ker_params, opt_params, interval, debug):
    """
    Creates segmentation masks.

    Parameters
    ----------
    channels : dict<str, 2darray>
        Frame's channel images by channel name.
    ker_params : KerParams
        Kernel parameters.
    opt_params : OptParams
        Optimization parameters.
    interval : str
        Frame name.
    debug : bool
        Is debug active.

    Returns
    -------
    chans : dict<str, 2darray>
        Masks by channel name.
    """
    chans = {}
    for channel in channels:
        if channel in seg_channel_types: # Create masks only for selected channels
            # Segment
            chans[channel] = pr.seg_phase(channels[channel], despeckle_size=1, dev_thresh=2, ker_params=ker_params, opt_params=opt_params, file_name=interval, debug=debug)
    return chans


def get_cells_con_comps(con_comps, debug=False):
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


def load_frame(interval, ker_params, opt_params, seq_paths, comps_dir, format, debug=False):
    """
    Loads frame.

    Parameters
    ----------
    interval : str
        Frame name.
    ker_params : KerParams
        Kernel parameters.
    opt_params : OptParams
        Optimization parameters.
    seq_paths : dict<str>
        Paths to sequence images.
    comps_dir : str
        Path to connected components target directory
    format : TitleFormat
        Image title format.
    debug : bool
        Is debug active

    Returns
    -------
    frame : Frame
        Loaded frame.
    """
    # Load channels
    images = create_stack(seq_paths[interval], opt_params=opt_params)
    # Segment
    masks = create_masks(images, ker_params=ker_params, opt_params=opt_params, interval=interval, debug=debug)
    # Generate and save connected components
    con_comp = 0
    for channel in seg_channel_types:
        if channel in masks:
            con_comp = pr.get_connected_components(masks[channel], grayscale=True, debug=debug)

    frame = fr.Frame(id=io_utils.extract_num(interval, format=format), title=interval, images=images, masks=masks,
                     con_comps=con_comp[1])
    print("Loaded frame {}\n".format(interval))

    return frame


def load_label_colors():
    for i in range(1, 2000):
        label_colors[i] = random.randint(1, 255)

def aggr_procs_tracked(result):
    """
    Aggregates results of processes loading the trackd frames.

    Parameters
    ----------
    result : Frame
        frame to aggregate

    """
    frame = result
    # Add frame to collection
    seq_frames[frame.id] = frame

    # Assign frame to global label
    for label in result.cells:
        if label not in cells_frames:
            cells_frames[label] = [frame.id]
        else:
            cells_frames[label].append(frame.id)

def aggr_procs(result):
    """
    Aggregates results of processes loading the frames.

    Parameters
    ----------
    result : Frame
        frame to aggregate

    """
    # Add frame to collection
    seq_frames[result.id] = result


def load_sequence(dir, ker_params, opt_params, comps_dir, format, debug=True, itr=500, procs=4):
    """
    Loads frame sequence.
    Parameters
    ----------
    dir : str
        Path to image sequence directory
    ker_params : KerParams
        Kernel parameters
    opt_params : OptParams
        Optimization parameters
    comps_dir : str
        Path to connected components target directory
    format : TitleFormat
        Image title format.
    debug : bool
        Is debug active
    itr : int
        Max number of iterations
    procs : int
        Number of concurrent processes to run

    """
    pool = mp.Pool(processes=procs)
    i = 0

    # Load all frames
    for interval in seq_paths:
        pool.apply_async(load_frame, args=(interval, ker_params, opt_params, seq_paths, comps_dir, format, debug), callback=aggr_procs)
        i += 1
        if i >= itr:
            break
    pool.close()
    pool.join()

    print("Finished loading sequence!\n") # DEBUG

def visualize_tracked_img(img, colors, name):
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


def track_sequence():
    """
    Track connected components via Lineage Mapper
    """
    args = []
    # args = ['inputDirectory', '', 'outputDirectory', ''] # Enables overriding arguments such as input/output paths (without changing config)
    subprocess.call(['java', '-jar', 'lib\Lineage_Mapper\Lineage_Mapper.jar'] + args, shell=True)


def get_tracked_cells(tracked_img, images, frame_id):
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
        first_elemnt = np.argwhere(tracked_img == label)[0]
        centroid = (first_elemnt[1], first_elemnt[0])  # TODO NOT REALLY CENTROID. USED FOR DEBUG WITH LABELS
        cell = cl.Cell(global_label=label, frame_label=label, pixel_values=channels_pixels, centroid=centroid)
        cells[label] = cell



    return cells


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


def load_tracked_frame(interval, tracked_paths, opt_params, seq_frames, format=mpar.TitleFormat.TRACK):
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
    if interval.startswith("trk-"):
        interval = interval[4:]
    frame_id = io_utils.extract_num(interval, format)
    frame = seq_frames[frame_id] # Original frame
    # Normalize channels
    for chan in frame.images:
        chan_img = frame.images[chan]
        frame.images[chan] = cv2.normalize(chan_img, None, 0, 255, cv2.NORM_MINMAX)
    tracked_img = load_tracked_mask(tracked_paths["trk-" + interval], opt_params)  # image after tracking
    cells = get_tracked_cells(tracked_img=tracked_img, images=frame.images, frame_id=frame_id)  # image's cell representation

    frame.tracked_img = tracked_img
    frame.cells = cells

    print("Loaded tracked frame {}\n".format(interval))
    return frame


def load_tracked_sequence(dir_tracked, format=mpar.TitleFormat.TRACK):
    """
    Loads frames after tracking.

    Parameters
    ----------
    dir_tracked : str
        Path to tracked images.
    format : TitleFormat
        Image title format.

    """
    pool = mp.Pool(processes=procs)
    # Load all frames
    tracked_paths = io_utils.load_paths(dir_tracked, format=format)
    for interval in tracked_paths:
        pool.apply_async(load_tracked_frame, args=(interval, tracked_paths, opt_params, seq_frames, format),
                         callback=aggr_procs_tracked)

    pool.close()
    pool.join()

def save_sequence_con_comps(comps_dir):
    """
    Save connected components generated from frame sequence.
    Parameters
    ----------
    comps_dir : str
        Path to save connected components.

    Returns
    -------

    """
    for frame in sorted(seq_frames):
        print("saving" + comps_dir + '\\' + str(seq_frames[frame].title) + ".tif")
        io_utils.save_img(seq_frames[frame].con_comps, comps_dir + '\\' + str(seq_frames[frame].title) + ".tif")


def stabilize_sequence(debug=False, procs=2, pad_pixels=25):
    aggr_shift_x = 0
    aggr_shift_y = 0
    shifts = {1: (0, 0)}
    crop_right = crop_left = crop_bottom = crop_top = 0

    # find optimal shifts
    for frame in sorted(seq_frames):
        if frame == 1:
            continue

        if debug:
            print("stabilizing frame {}".format(frame))

        # step 2 - find shift vector in pixels for connected components
            print("finding optimal shift")
        shift_cost = {}
        phase_chan = [x for x in seq_frames[frame].images if x in seg_channel_types][0]
        phase = np.float32(seq_frames[frame].images[phase_chan])
        prev_phase = np.float32(seq_frames[frame - 1].images[phase_chan])
        rows, cols = phase.shape

        final_shift = calc_shift(prev_phase, phase, max_shift=25, procs=4)
        final_shift = (final_shift[0] + aggr_shift_x, final_shift[1] + aggr_shift_y)
        shifts[frame] = final_shift
        print("aggregated shift = {}".format(final_shift))

        # update cropping
        if final_shift[0] > 0:
            crop_left = max(crop_left, final_shift[0])
        else:
            crop_right = max(crop_right, abs(final_shift[0]))
        if final_shift[1] > 0:
            crop_top = max(crop_top, final_shift[1])
        else:
            crop_bottom = max(crop_bottom, abs(final_shift[1]))

        aggr_shift_x, aggr_shift_y = final_shift

    print("cropping: top: {}, bottom: {}, left: {}, right: {}".format(crop_top, crop_bottom, crop_left, crop_right))

    # shift and crop frames
    for frame in sorted(seq_frames):
        shifted = translate_img(seq_frames[frame].con_comps, shifts[frame][0], shifts[frame][1])
        seq_frames[frame].con_comps = crop_img(shifted, crop_right, crop_left, crop_top, crop_bottom)
        if debug:
            plt.imshow(seq_frames[frame].con_comps)
            plt.show()

        # shift aux channels
        for channel in seq_frames[frame].images:
            if channel in aux_channel_types:
                shifted = translate_img(seq_frames[frame].images[channel], shifts[frame][0], shifts[frame][1])
                seq_frames[frame].images[channel] = crop_img(shifted, crop_right, crop_left, crop_top, crop_bottom)
                if debug:
                    plt.imshow(seq_frames[frame].images[channel])
                    plt.show()

def calc_shift(prev: np.ndarray, cur: np.ndarray, max_shift: int = 1, procs=2) -> Tuple[int, int]:
    diff.clear()
    pool = mp.Pool(processes=procs)

    coords = [(x,y) for x in range(-max_shift, max_shift + 1) for y in range(-max_shift, max_shift + 1)]
    coords = np.array_split(coords, procs)
    for i in range(procs):
        pool.apply_async(calc_diff_shifted, args=(prev, cur, coords[i]), callback=diff_result_cb)

    pool.close()
    pool.join()
    opt_shift = min(diff, key=diff.get)
    print("optimal shift = {}".format(opt_shift))
    print("diff = {}".format(diff[opt_shift]))
    return opt_shift

def calc_diff_shifted(prev, cur, coords):
    costs = {}
    for (x, y) in coords:
        shifted = np.roll(cur, y, axis=0)
        shifted = np.roll(shifted, x, axis=1)
        costs[x,y] = np.sum(np.sum(np.abs(shifted - prev)))
    opt_shift = min(costs, key=costs.get)
    return (opt_shift, costs[opt_shift])


def diff_result_cb(result):
    diff[result[0]] = result[1]


def translate_img(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    if shift_x == 0 and shift_y == 0:
        return img
    rows, cols = img.shape

    shifted = np.roll(img, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)

    # nullify overflowing elements
    if shift_x > 0:
        shifted[:, :shift_x] = 0
    else:
        shifted[:, cols + shift_x:cols] = 0

    if shift_y > 0:
        shifted[:shift_y, :] = 0
    else:
        shifted[rows + shift_y: rows, :] = 0

    return shifted


def crop_img(img: np.ndarray, right: int, left: int, top: int, bottom: int) -> np.ndarray:
    rows, cols = img.shape

    img[:, :left] = 0

    img[:, cols - right: cols] = 0

    img[:top, :] = 0

    img[rows - bottom: rows, :] = 0

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
    for channel in channels:
        frame_bg = {}  # dictionary <img, float, float> that holds the current frame channel + mean background

        # iterate over first frame identified cells
        for cell in cells_frames:
            if cell != 0:  # skip background label
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


def check_changed(frame_id, frame_stat, label, channel, thresh_change=0.5):  # TODO change threshold

    # calculate cell average intensity, if it is substantially larger than background -> decide cell is colored
    cell = seq_frames[frame_id].cells[label]

    pixels = cell.pixel_values[channel]
    # cell_mean = np.mean(cell.pixel_values[channel])
    cell_area = pixels.size
    cell_colored = np.sum(pixels) / 255
    #cell_colored = pixels.size
    cell_intensity = cell_colored / cell_area
    # cell_intensity = cell_colored

    if cell_intensity > thresh_change:
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
                        cv2.putText(dbg_frame, str(cell), seq_frames[frame_id].cells[cell].centroid,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 190, 2)
            path = dir + "\\" + channel + str(frame_id) + ".png"
            thresh_path = dir + "\\" + channel + str(frame_id) + "_THRESH.png"
            io_utils.save_img(dbg_frame, path, uint8=True)
            io_utils.save_img(seq_frames[frame_id].images[channel], thresh_path, uint8=True)

            # Visualize stack
            for channel in seg_channel_types:
                frame = seq_frames[frame_id]
                images = frame.images
                masks = frame.masks
                tracked = frame.tracked_img
                if channel in masks:
                    b = cv2.normalize(tracked, None, 0, 255, cv2.NORM_MINMAX)
                    g = cv2.normalize(images["fitc"], None, 0, 100, cv2.NORM_MINMAX)
                    r = cv2.normalize(images["PI"], None, 0, 255, cv2.NORM_MINMAX)
                    vis = np.dstack((b, g, r))  # TODO change so it will work for seq_nec
                    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imwrite(comps_dir + '\\' + 'vis\\' + str(frame_id) + ".tif", vis)


if __name__ == "__main__":

    """img_to_align = img_utils.load_img("images\\L136\\A2\\4\\L136_phase_A2_4_2018y02m12d_10h30m.tif", 0.5, True, False)
    img_ref = img_utils.load_img("images\\L136\\A2\\4\\L136_phase_A2_4_2018y02m12d_10h45m.tif", 0.5, True, False)
    pr.align_img(img_to_align,img_ref)"""

    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=1, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)
    dir = "images\\L136\\A2\\4"
    comps_dir = "images\\L136\\A2\\4\\concomps"

    """dir = "images\\seq_nec"
    comps_dir = "images\\seq_nec\\concomps"""

    # load_tracked_masks("images\\seq_nec\\tracked")
    print("Started sequence loading\n")

    seq_paths = io_utils.load_paths(dir)
    iterations = 20
    procs = 2
    debug = False
    file_format = mpar.TitleFormat.DATE

    seq_paths = io_utils.load_paths(dir, format=file_format)

    load_sequence(dir, ker_params=ker_params, opt_params=opt_params,comps_dir=comps_dir, debug=debug, itr=iterations, format=mpar.TitleFormat.TRACK, procs=procs)

    print("Finished sequence Loading\n")

    print("Started sequence stabilization\n")

    stabilize_sequence(True, procs);

    print("Finished sequence stabilization\n")

    print("Saving connected components...\n")

    save_sequence_con_comps(comps_dir)
    #exit()

    print("Started sequence tracking\n")

    track_sequence()

    print("Finished sequence tracking\n")

    print("Started loading tracked sequence\n")

    load_tracked_sequence(dir_tracked="images\\L136\\A2\\4\\concomps\\track", format=mpar.TitleFormat.TRACK)

    print("Finished loading tracked sequence\n")

    analyze_channels(["fitc", "PI"])

    print("Finished analyze_channels\n")

    debug_channels("dbg\\L136\\A2\\4", ["fitc", "PI"])

    print("Finished debug_channels\n")

    # load_tracked_masks("images\\seq_nec\\concomps\\track", opt_params)

# save_con_comps()
