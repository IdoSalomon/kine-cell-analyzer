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

seq_paths = {} # Paths to all sequence images
from prec_params import KerParams, OptParams

seq_frames = {} # dictionary <int,Frame> that holds all frames in sequence by number

label_colors = {} # dictionary <int, int> that holds constant colors for each labelled cells

channel_types = ["GFP", "FITC", "fitc", "pi", "PI", "PHASE", "phase", "TxRed", "TRANS", "trans"] # different channels in sequence TODO turn to user input

seg_channel_types = ["PHASE", "TRANS", "phase"] # different channels in sequence to segment TODO turn to user input

aux_channel_types = ["GFP", "TxRed", "fitc", "FITC", "PI", "pi"]

cells_frames = {} # dictionary <int, list<int>> that holds IDs of frame in which the cell appeared

cells_trans = {} # dictionary <int, dict<str, int>> that holds for each cell ID the transformation frame ID by channel


def create_stack(chan_paths, opt_params):
    channels = {}
    # Scan all channels
    for chan_path in chan_paths:
        # Check if channel is monitored
        for chan_type in channel_types:
            # Add relevant image to channel
            if chan_type in chan_path:
                channels[chan_type] = img_utils.load_img(chan_path, opt_params.img_scale, False, False)
                if chan_type in aux_channel_types:
                    ext.seg_aux_channels(channels[chan_type], chan_type) # FIXME Already float
                break

    return channels


def create_masks(channels, ker_params, opt_params, interval, debug):
    chans = {}
    for channel in channels:
        if channel in seg_channel_types:
            chans[channel] = pr.seg_phase(channels[channel], despeckle_size=1, dev_thresh=2, ker_params=ker_params, opt_params=opt_params, file_name=interval, debug=debug)
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


def load_frame(interval, ker_params, opt_params, seq_paths, comps_dir, format, debug=False):
    # Load channels
    images = create_stack(seq_paths[interval], opt_params=opt_params)
    # Segment
    masks = create_masks(images, ker_params=ker_params, opt_params=opt_params, interval=interval, debug=debug)
    # Generate and save connected components
    con_comp = 0
    for channel in seg_channel_types:
        if channel in masks:
            con_comp = pr.get_connected_components(masks[channel], grayscale=True, debug=debug)
            b = cv2.normalize(con_comp[1], None, 0, 255, cv2.NORM_MINMAX)
            g = cv2.normalize(images["fitc"], None, 0, 255, cv2.NORM_MINMAX)
            r = cv2.normalize(images["PI"], None, 0, 255, cv2.NORM_MINMAX)
            vis = np.dstack((b, g, r)) # TODO change so it will work for seq_nec
            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(comps_dir + '\\' + 'vis\\' + interval + ".tif", vis)

    frame = fr.Frame(id=io_utils.extract_num(interval,format=format), title=interval, images=images, masks=masks, con_comps=con_comp[1])
    print("Loaded frame {}\n".format(interval))

    return frame


def load_label_colors():
    for i in range(1, 2000):
        label_colors[i] = random.randint(1, 255)


def load_sequence(dir, ker_params, opt_params, comps_dir, format, debug=True, itr=500):
    i = 0
    for interval in seq_paths:
        frame = load_frame(interval, ker_params=ker_params, opt_params=opt_params, seq_paths=seq_paths, debug=debug, comps_dir=comps_dir, format=format)
        seq_frames[frame.id] = frame

        i += 1
        if i >= itr:
            break

    print("Finished loading sequence!\n") # DEBUG

def save_con_comps(dir):
    print("todo")
    # TODO

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


def load_tracked_masks(dir, format=mpar.TitleFormat.SCENE): # TODO DANIEL
    print("load_tracked_masks\n")
    tracked_paths = io_utils.load_paths(dir, format)
    load_label_colors()
    for tracked_path in tracked_paths:
        tracked_img = img_utils.load_img(dir + "\\" + tracked_path, 1, False, False, float=False, normalize=False)
        visualize_tracked_img(tracked_img, label_colors, io_utils.extract_name(tracked_path, format=format))


def track_sequence():
    """
    Calls Lineage Mapper to track connected components
    """
    args = []
    #args = ['inputDirectory', '', 'outputDirectory', ''] # Enables overriding arguments such as input/output paths (without changing config)
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
        centroid = (first_elemnt[1], first_elemnt[0]) # TODO NOT REALLY CENTROID. USED FOR DEBUG WITH LABELS
        cell = cl.Cell(global_label=label, frame_label=label, pixel_values=channels_pixels, centroid=centroid)
        cells[label] = cell

        # Assign frame to global label
        if label not in cells_frames:
            cells_frames[label] = [frame_id]
        else:
            cells_frames[label].append(frame_id)

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


def load_tracked_frame(interval, tracked_paths, opt_params, seq_paths, format=mpar.TitleFormat.TRACK):
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
    images = create_stack(seq_paths[interval], opt_params=opt_params) # original channels
    tracked_img = load_tracked_mask(tracked_paths["trk-" + interval], opt_params) # image after tracking
    cells = get_tracked_cells(tracked_img=tracked_img, images=images, frame_id=frame_id) # image's cell representation

    frame = seq_frames[frame_id]
    frame.tracked_img = tracked_img
    frame.cells = cells

    print("Loaded tracked frame {}\n".format(interval))
    return frame

def load_tracked_sequence(dir_tracked, format=mpar.TitleFormat.TRACK):
    tracked_paths = io_utils.load_paths(dir_tracked, format=format)
    for interval in tracked_paths:
        frame = load_tracked_frame(interval, opt_params=opt_params, seq_paths=seq_paths, tracked_paths=tracked_paths, format=format)

        seq_frames[frame.id] = frame

    print("Finished loading sequence!\n")  # DEBUG

def save_sequence_con_comps(comps_dir):
    for frame in sorted(seq_frames):
        print("saving" + comps_dir + '\\' + str(seq_frames[frame].title) + ".tif")
        io_utils.save_img(seq_frames[frame].con_comps, comps_dir + '\\' + str(seq_frames[frame].title) + ".tif")  # TODO Remove
        # # expand aux. channels + connected components
        # if debug:
        #     print("expanding connected components")
        # seq_frames[frame].con_comps = iu.expand_img(seq_frames[frame].con_comps, pad_pixels=pad_pixels)
        #
        # for channel in seq_frames[frame].images:
        #     if channel in aux_channel_types:
        #         if debug:
        #             print("expanding {} channel".format(channel))
        #         seq_frames[frame].images[channel] = iu.expand_img(seq_frames[frame].images[channel], pad_pixels=pad_pixels)

def stabilize_sequence(debug = False, pad_pixels=30):
    aggr_shift_x = 0
    aggr_shift_y = 0
    for frame in sorted(seq_frames):
        if frame == 1:
            continue

        if debug:
            print("stabilizing frame {}".format(frame))

        # step 2 - find shift vector in pixels for connected components
        if debug:
            print("finding optimal shift")
        shift_cost = {}
        phase_chan = [x for x in seq_frames[frame].images if x in seg_channel_types][0]
        phase = np.float32(seq_frames[frame].images[phase_chan])
        prev_phase = np.float32(seq_frames[frame-1].images[phase_chan])
        rows, cols = phase.shape

        # for x in range(-pad_pixels, pad_pixels):
        #     for y in range(-pad_pixels, pad_pixels):
        #         if x==29 and y == 29:
        warp_matrix = []
        pre_shift = (0,0)
        is_found = False
        try:
            warp_matrix = pr.align_img(phase, prev_phase)
            print("align_img successful, no need to shift before calling")
        except:
            print("align_img failed, trying to shift before calling")
            for i in range(5, 35, 5):
                if is_found:
                    break
                shifts = [x for x in itertools.product([0, i, -i], repeat=2)]
                for shift in shifts:
                    print("trying shift {}".format(shift))
                    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                    shifted =  cv2.warpAffine(phase ,M , (cols, rows), flags=cv2.WARP_INVERSE_MAP)
                    try:
                        warp_matrix = pr.align_img(shifted, prev_phase)
                    except:
                        continue
                    is_found = True
                    pre_shift = shift
                    print("pre-shift is {}".format(pre_shift))
                    break

        shiftx, shifty = (int(round(warp_matrix[0][2])), int(round(warp_matrix[1][2])))
        print("align_img return shift: {}".format((warp_matrix[0][2] , warp_matrix[1][2])))
        final_shift = (int(round(warp_matrix[0][2] + pre_shift[0] + aggr_shift_x)) ,int(round(warp_matrix[1][2] + pre_shift[1] + aggr_shift_y)))
        print("final shift: {}".format(final_shift))

        warp_matrix[0][2] = aggr_shift_x = final_shift[0]
        warp_matrix[1][2] = aggr_shift_y =  final_shift[1]

        # shift connected components
        seq_frames[frame].con_comps = cv2.warpAffine(np.float32(seq_frames[frame].con_comps) ,warp_matrix , (cols, rows), flags=cv2.WARP_INVERSE_MAP)
        plt.imshow(seq_frames[frame].con_comps)
        plt.show()

        # shift aux channels
        for channel in seq_frames[frame].images:
            if channel in aux_channel_types:
                seq_frames[frame].images[channel] = cv2.warpAffine(np.float32(seq_frames[frame].images[channel]) ,warp_matrix , (cols, rows), flags=cv2.WARP_INVERSE_MAP)
                if debug:
                    plt.imshow(seq_frames[frame].images[channel])
                    plt.show()

        #shift_cost[x, y] = sum(con_comps - shifted_xy)
        #print(min(shift_cost.values()))




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


def check_changed(frame_id, frame_stat, label, channel, thresh_change=0.3): # TODO change threshold

    # calculate cell average intensity, if it is substantially larger than background -> decide cell is colored
    cell = seq_frames[frame_id].cells[label]

    pixels = cell.pixel_values[channel]
    # cell_mean = np.mean(cell.pixel_values[channel])
    cell_area = pixels.size
    #cell_colored = np.sum(pixels) / 256
    cell_colored = pixels[pixels > 15].size
    cell_intensity = cell_colored / cell_area
    #cell_intensity = cell_colored

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
                        cv2.putText(dbg_frame, str(cell), seq_frames[frame_id].cells[cell].centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 190, 2)
            path = dir + "\\" + channel + str(frame_id) + ".png"
            thresh_path = dir + "\\" + channel + str(frame_id) + "_THRESH.png"
            io_utils.save_img(dbg_frame, path, uint8=True)
            io_utils.save_img(seq_frames[frame_id].images[channel], thresh_path, uint8=True)


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
    iterations = 3
    debug = False
    file_format = mpar.TitleFormat.DATE

    seq_paths = io_utils.load_paths(dir, format=file_format)

    load_sequence(dir, ker_params=ker_params, opt_params=opt_params,comps_dir=comps_dir, debug=debug, itr=iterations, format=mpar.TitleFormat.TRACK)

    print("Finished sequence Loading\n")

    print("Started sequence stabilization\n")

    stabilize_sequence(True);

    print("Finished sequence stabilization\n")

    print("Saving connected components...\n")

    save_sequence_con_comps(comps_dir)

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


#save_con_comps()
