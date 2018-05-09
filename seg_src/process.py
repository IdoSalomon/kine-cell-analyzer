import cv2
import numpy as np
import debug_utils as dbg
import img_utils
import io_utils
import prec_sparse as ps
import img_utils as iu
from prec_params import KerParams, OptParams


def gen_phase_mask(restored, orig_img, despeckle_size=0, filter_size=0, file_name="gen_phase_mask", debug=True):
    dbgImgs = []
    if debug:
        dbgImgs += [(restored, 'initial')]

    #normailize image to unit8 range: 0-255
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)
    orig_img = cv2.normalize(orig_img, None, 0, 255, cv2.NORM_MINMAX)

    restored = np.uint8(restored)
    orig_img = np.uint8(orig_img)

    kernel = np.ones((3, 3), np.uint8)

    # step 1 - threshold
    tmp, threshold = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY)
    if debug:
        dbgImgs += [(threshold, 'step 1 - threshold')]

    # step 2 - filter far/dead cells
    filtered = pre_filter_far_cells(threshold, despeckle_size, debug)
    if debug:
        dbgImgs += [(filtered, 'step 2 - pre-filter')]

    # step 3 - dilate
    dilated = cv2.dilate(filtered,kernel,iterations=1)
    #dilated = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        dbgImgs += [(dilated, 'step 3 - dilate')]

    # step 4 - filter far/dead cells
    filtered = filter_far_cells(dilated, filter_size, debug)
    if debug:
        dbgImgs += [(filtered, 'step 4 - filter')]

    filtered = filtered.astype(np.uint8)
    # step 5 - borders
    borders = cv2.Canny(filtered, 50, 200)
    if debug:
        dbgImgs += [(borders, 'step 5 - borders')]

    # step 6 - compare with original
    compare = np.copy(orig_img)
    compare[np.nonzero(borders == 255)] = 255

    if debug:
        dbgImgs += [(compare, 'step 6 - compare')]
        dbgImgs += [(orig_img, 'original image')]
        dbg.save_debug_fig(dbgImgs, file_name, zoom=5)

    return filtered

def get_connected_components(img, dst_path, grayscale=True, debug=True):

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    img = np.uint8(img)

    connectivity = 4
    labels = cv2.connectedComponentsWithStats(img, connectivity=connectivity)

    io_utils.save_img(labels[1], dst_path)  # TODO Remove

    return labels

def filter_far_cells(thresh, dev_thresh=1, debug = True):
    # Find connected components
    connectivity = 4
    thresh = thresh.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=connectivity)
    init_components = nb_components
    filtered_components = init_components

    # Find new cells area
    area = np.array([stats[i,cv2.CC_STAT_AREA] for i in range(1, len(stats))])
    mean = np.mean(area)
    std = np.std(area)

    filter_size = mean - dev_thresh * std # dev_thresh should be 1 for round, 1.1 for long

    # Remove background
    sizes = stats[1:, -1]
    nb_components -= 1
    min_size = filter_size

    # Filter
    filtered = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_components -= 1
            filtered[output == i + 1] = 255

    if debug:
        print('filtered {} out of {} segmented particles. {} cells remain.'.format(filtered_components, init_components, init_components - filtered_components))

    return filtered


def pre_filter_far_cells(thresh, despeckle_size=0, debug=True):
    # Find connected components
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=connectivity)

    init_components = nb_components
    filtered_components = init_components

    # Find cells area
    area = np.array([stats[i,cv2.CC_STAT_AREA] for i in range(1, len(stats))]) # TODO check why cell 0 area is large

    # Remove background
    sizes = stats[1:, -1];
    nb_components -= 1

    # Minimum size of particles to keep
    min_size = despeckle_size # should be 3 for long, 9-10 for round

    # Filter
    filtered = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_components -= 1
            filtered[output == i + 1] = 255

    if debug:
        print('filtered {} out of {} segmented particles. {} cells remain.'.format(filtered_components, init_components, init_components - filtered_components))

    return filtered


def process_aux_channels(img, despeckle_size=1, kernel=np.ones((2, 2), np.uint8), dev_thresh=0):
    kernel = np.ones((2, 2), np.uint8)

    # step 1 - filter noise
    filtered = pre_filter_far_cells(img, despeckle_size=despeckle_size)

    # step 2 - dilate
    dilated = cv2.dilate(filtered, kernel, iterations=1)
    filtered = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    # step 3 - filter far/dead cells
    filtered = filter_far_cells(filtered, dev_thresh=dev_thresh)

    return filtered

def seg_phase(img, opt_params=0, ker_params=0, despeckle_size=1, dev_thresh=0, file_name=0, debug=False):
    additive_zero = ps.prec_sparse(img, opt_params, ker_params, debug)[:, :, 0]
    ker_params_first = KerParams(ring_rad=ker_params.ring_rad, ring_wid=ker_params.ring_wid, ker_rad=ker_params.ker_rad + 1, zetap=ker_params.zetap, dict_size=ker_params.dict_size)
    next = ps.prec_sparse(img, opt_params, ker_params_first, True)
    next_first = next[:, :, 1]
    next = next[:, :, 0]

    additive_zero = cv2.add(additive_zero, next)
    post_proc = gen_phase_mask(additive_zero, img, despeckle_size=despeckle_size, filter_size=dev_thresh, file_name=file_name)

    next_first = cv2.normalize(next_first, None, 0, 255, cv2.NORM_MINMAX)

    tmp, next_first_mask = cv2.threshold(next_first, 1, 255, cv2.THRESH_BINARY)

    #next_first_mask = cv2.normalize(next_first_mask, None, 0, 255, cv2.NORM_MINMAX)


    next_first_mask = pre_filter_far_cells(np.uint8(next_first_mask), despeckle_size=1, debug=debug)
    next_first_mask = cv2.morphologyEx(next_first_mask, cv2.MORPH_CLOSE, (3, 3), iterations=3)


    #next_first_mask = filter_far_cells(next_first_mask, dev_thresh=dev_thresh, debug=debug)

    sub_next = cv2.subtract(post_proc, np.uint8(next_first_mask))

    sub_next = pre_filter_far_cells(np.uint8(sub_next), despeckle_size=3, debug=debug)
    sub_next = filter_far_cells(np.uint8(sub_next), dev_thresh=1.8, debug=debug)

    io_utils.save_img(sub_next, "dbg\\test_sub.png")  # TODO Remove
    io_utils.save_img(post_proc, "dbg\\test_proc.png")  # TODO Remove
    io_utils.save_img(next_first_mask, "dbg\\test_next_first_mask.png")  # TODO Remove
    io_utils.save_img(next_first, "dbg\\test_next_first.png")  # TODO Remove


    return sub_next


def gen_gfp_mask(raw_threshold, orig_img, debug=True):

    """img = np.uint8(img)
    raw_threshold = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    tmp, threshold = cv2.threshold(raw_threshold, 0, 255, cv2.THRESH_BINARY)

    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 2)
    kernel = np.ones((2, 2), np.uint8)  # small kernel for closing gaps in cells
    kernel_closing = np.ones((4, 4), np.uint8)  # kernel for clearing noise, small "noise" cell are removed(phase, red)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closing, iterations=3)

    sure_bg = cv2.dilate(closing, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

    ret, sure_fg = cv2.threshold(dist_transform, 0.005 * dist_transform.max(), 255, 0)

    # Finding unknown region green
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    img[markers == 1] = [255, 255, 255]
    img[markers != 1] = [0, 0, 0]

    img[markers == -1] = [0, 0, 0]

    img = cv2.dilate(img, kernel, iterations=5)

    if True:
        dbg.save_debug_fig(dbgImgs, 'GFP mask', zoom=5)"""


def seg_gfp(img, opt_params=0, ker_params=0, debug=True):
   """ #TODO remove params
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=2, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)

    res_img = ps.prec_sparse(img, opt_params, ker_params, debug)
    red_channel = res_img[:, :, 0]
    return gen_gfp_mask(red_channel, img, True)"""

if __name__ == "__main__":
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)
    # For Bovine aortic endothelial cell(BAEC)
    #ker_params = KerParams(ring_rad=4000, ring_wid=800, ker_rad=5, zetap=0.8, dict_size=20)

    #ker_params = KerParams(ring_rad=10, ring_wid=0.05, ker_rad=10, zetap=0.8, dict_size=20)

    #seg_phase(iu.load_img("images\\seq_nec\\Scene1Interval043_PHASE.png", 0.3, False, False), ker_params=ker_params, opt_params=opt_params, despeckle_size=3, filter_size=0, file_name="gen_phase_mask.png")
    seg_phase(iu.load_img("images\\seq_nec\\Scene1Interval029_PHASE.png", 0.5, False, False), ker_params=ker_params, opt_params=opt_params, despeckle_size=3, dev_thresh=0, file_name="gen_phase_mask.png")
    #seg_phase(iu.load_img("images\\small.png", 1, False, False), ker_params=ker_params, opt_params=opt_params, despeckle_size=3, filter_size=0, file_name="gen_phase_mask.png")
