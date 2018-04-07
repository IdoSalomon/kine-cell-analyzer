import cv2
import numpy as np
import debug_utils as dbg
import prec_sparse as ps
import img_utils as iu
from prec_params import KerParams, OptParams


def gen_phase_mask(raw_threshold, orig_img, file_name="gen_phase_mask", debug=True):
    dbgImgs = []
    if debug:
        dbgImgs += [(raw_threshold, 'initial')]

    #normailize image to unit8 range: 0-255
    raw_threshold = cv2.normalize(raw_threshold, None, 0, 255, cv2.NORM_MINMAX)
    orig_img = cv2.normalize(orig_img, None, 0, 255, cv2.NORM_MINMAX)
    raw_threshold = np.uint8(raw_threshold)
    orig_img = np.uint8(orig_img)

    kernel = np.ones((3, 3), np.uint8)

    # step 1 - threshold
    tmp, threshold = cv2.threshold(raw_threshold, 0, 255, cv2.THRESH_BINARY)
    if debug:
        dbgImgs += [(threshold, 'step 1 - threshold')]

        # step 3 - filter far/dead cells
    filtered = filter_far_cells(threshold, debug)
    if debug:
        dbgImgs += [(filtered, 'step 3 - filter')]

    # step 2 - dilate
    dilated = cv2.dilate(filtered,kernel,iterations = 1)
    if debug:
        dbgImgs += [(dilated, 'step 2 - dilate')]


    # step 4 - borders
    borders = cv2.Canny(filtered, 100, 200)
    if debug:
        dbgImgs += [(borders, 'step 4 - borders')]

    # step 5 - compare with original
    compare = np.copy(orig_img)
    compare[np.nonzero(borders == 255)] = 255
    if debug:
        dbgImgs += [(compare, 'step 5 - compare')]
        dbgImgs += [(orig_img, 'original image')]
        dbg.save_debug_fig(dbgImgs, file_name, zoom=5)


def colorConnectedComponents(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    img = np.uint8(img)

    ret, labels = cv2.connectedComponents(img, connectivity=4)

    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

def filter_far_cells(thresh, debug = True):

    #list cells
    connectivity = 4
    con_comps = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    labels = con_comps[1]
    stats = con_comps[2]

    #iterate over components and store cells area
    area = np.array([stats[i,cv2.CC_STAT_AREA] for i in range(1, len(stats))]) # TODO check why cell 0 area is large
    mean = np.mean(area)
    std = np.std(area)

    # Descpeckle
    speckles = [i for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] < 4]
    mask = np.zeros((len(labels), len(labels[0])))
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i,j] in speckles:
                mask[i, j] = 1

    thresh[np.nonzero(mask == 1)] = 0

    # Filter
    con_comps = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    labels = con_comps[1]
    stats = con_comps[2]

    small = [i for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] < mean - std]
    mask = np.zeros((len(labels), len(labels[0])))
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i,j] in small:
                mask[i, j] = 1

    filtered = np.copy(thresh)

    filtered[np.nonzero(mask == 1)] = 0

    if debug:
        print('filtered {} out of {} segmented particles. {} cells remain.'.format(len(small) + len(speckles), len(labels), len(labels) - len(small) - len(speckles)))

    return filtered


def seg_phase(img, opt_params=0, ker_params=0, file_name=0, debug=True):
    # TODO remove params
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)
    res_img = ps.prec_sparse(img, opt_params, ker_params, debug)
    red_channel = res_img[:, :, 0]
    return gen_phase_mask(red_channel, img, file_name=file_name)


def gen_gfp_mask(raw_threshold, orig_img, debug=True):
    dbgImgs = []
    #normailize image to unit8 range: 0-255
    raw_threshold = cv2.normalize(raw_threshold, None, 0, 255, cv2.NORM_MINMAX)
    orig_img = cv2.normalize(orig_img, None, 0, 255, cv2.NORM_MINMAX)
    raw_threshold = np.uint8(raw_threshold)
    orig_img = np.uint8(orig_img)

    # step 1 - threshold
    tmp, threshold = cv2.threshold(raw_threshold, 0, 255, cv2.THRESH_BINARY)

    if debug:
        dbgImgs += [(threshold, 'step 1 - threshold')]

    # noise removal - remove small
    kernel_open = np.ones((2, 2), np.uint8)  # small kernel for closing gaps in cells
    kernel_close = np.ones((1, 1), np.uint8)  # small kernel for closing gaps in cells

    dilated = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel_open, iterations=2)  # dialate + erode
    dilated = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel_close, iterations=2)  # dialate + erode

    if debug:
        dbgImgs += [(dilated, 'step 3 - filter')]

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

    img = cv2.dilate(img, kernel, iterations=5)"""

    if True:
        dbg.save_debug_fig(dbgImgs, 'GFP mask', zoom=5)


def seg_gfp(img, opt_params=0, ker_params=0, debug=True):
    #TODO remove params
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=2, epsilon=3, gamma=3, img_scale=0.5,
                           max_itr=100, opt_tolr=np.finfo(float).eps)

    res_img = ps.prec_sparse(img, opt_params, ker_params, debug)
    red_channel = res_img[:, :, 0]
    return gen_gfp_mask(red_channel, img, True)

if __name__ == "__main__":
    seg_gfp(iu.load_img("images\\seq_apo\\Scene1Interval158_GFP.tif", 0.5, False, False))
