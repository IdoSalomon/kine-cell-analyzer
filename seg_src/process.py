import cv2
import cython as cython
import numpy as np
import pandas as pd

import debug_utils as dbg
import img_utils
import io_utils
import prec_sparse as ps
import img_utils as iu
from prec_params import KerParams, OptParams
from scipy import ndimage
from sklearn.cluster import KMeans

"""
%load_ext cython
%%cython -a
import cython

@cython.boundscheck(False)
cpdef unsigned char[:, :] threshold_fast(int T, unsigned char[:, :] image):
    # set the variable extension types
    cdef int x, y, w, h

    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image[y, x] >= T else 0

    # return the thresholded image
    return image
"""

def find_thresh_aux(img):
    thresh_color = np.percentile(img, 99.9)
    thresh_tri, triangle = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    if not img_utils.is_noisy(triangle):
        thresh_color = thresh_tri
    return thresh_color

def get_gradient(img):
    # Calculate the x and y gradients using Sobel operator
    grad_x = np.float32(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    grad_y = np.float32(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    return grad

def align_img(img_to_align, img_ref, thresh=1e-7, warp_mode=cv2.MOTION_TRANSLATION):
    img_to_align = np.copy(img_to_align)
    # Find the width and height of the image
    height, width = img_ref.shape

    # Set the warp matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32) # Homography
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32) # Affine

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, thresh)

    # Find warp matrix for alignment
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(img_ref), get_gradient(img_to_align),
                                             warp_matrix, warp_mode, criteria)

    # Warp
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use Perspective warp when the transformation is a Homography
        img_aligned = cv2.warpPerspective(img_to_align, warp_matrix, (width, height),
                                     flags=cv2.WARP_INVERSE_MAP)
    else:
        # Use Affine warp when the transformation is not a Homography
        img_aligned = cv2.warpAffine(img_to_align, warp_matrix, (width, height),
                                     flags=cv2.WARP_INVERSE_MAP)

    # Save final output
    # cv2.imwrite("dbg\\ColorImage.png", cv2.normalize(img_ref, None, 0, 255, cv2.NORM_MINMAX))
    #  cv2.imwrite("dbg\\ColorAlignedImage.png", cv2.normalize(img_aligned, None, 0, 255, cv2.NORM_MINMAX))

    return warp_matrix

def padding_for_kernel(kernel):
    """ Return the amount of padding needed for each side of an image.

    For example, if the returned result is [1, 2], then this means an
    image should be padded with 1 extra row on top and bottom, and 2
    extra columns on the left and right.
    """
    # Slice to ignore RGB channels if they exist.
    image_shape = [kernel[0], kernel[1]]
    # We only handle kernels with odd dimensions so make sure that's true.
    # (The "center" pixel of an even number of pixels is arbitrary.)
    assert all((size % 2) == 1 for size in image_shape)
    return [(size - 1) // 2 for size in image_shape]

def remove_padding(image, kernel):
    inner_region = []  # A 2D slice for grabbing the inner image region
    for pad in padding_for_kernel(kernel):
        slice_i = slice(None) if pad == 0 else slice(pad, -pad)
        inner_region.append(slice_i)
    return image[inner_region]

def add_padding(image, kernel):
    h_pad, w_pad = padding_for_kernel(kernel)
    return np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=0)

def window_slice(center, kernel):
    r, c = center
    r_pad, c_pad = padding_for_kernel(kernel)
    # Slicing is (inclusive, exclusive) so add 1 to the stop value
    return [slice(r-r_pad, r+r_pad+1), slice(c-c_pad, c+c_pad+1)]


def find_borders_naive(labels, ker):
    label_img = labels[1]
    stats = labels[2]

    # Get original image dimensions
    h = label_img.shape[0]
    w = label_img.shape[1]

    # Pad labels image (for sliding window)
    pad_labels = add_padding(label_img, ker)
    # Get padding dimensions (for index mapping)
    pad_h = (int)((pad_labels.shape[0] - h) / 2)
    pad_w = (int)((pad_labels.shape[1] - w) / 2)

    borders = np.zeros((h,w))

    # Loop over the image, pixel by pixel
    for y in range(pad_h, h + pad_h):
        for x in range(pad_w, w + pad_w):
            if pad_labels[y, x] == 0:
                ker_slice = pad_labels[window_slice(center=(y,x), kernel=ker)] # window
                uniques = np.unique(ker_slice)
                uniques = uniques[uniques != 0] # remove background label
                num_uniques = uniques.size
                pos = False
                # Avoid decimating sparsely detected cells
                for i in range(num_uniques):
                    if stats[uniques[i], cv2.CC_STAT_AREA] > 13:
                        pos = True
                # Add to mask if is border between large connected components
                if num_uniques > 1 and pos:
                    borders[y - pad_h, x - pad_w] = 255

    return borders

def find_borders(labels, ker):
    label_img = labels
    rows = []

    # Get original image dimensions
    h = label_img.shape[0]
    w = label_img.shape[1]

    # Pad labels image (for sliding window)
    pad_labels = add_padding(label_img, ker)
    # Get padding dimensions (for index mapping)
    pad_h = (int)((pad_labels.shape[0] - h) / 2)
    pad_w = (int)((pad_labels.shape[1] - w) / 2)

    borders = np.zeros((h,w))



    # Loop over the image, pixel by pixel
    for y in range(pad_h, h + pad_h):
        for x in range(pad_w, w + pad_w):
            if pad_labels[y, x] == 0:
                ker_slice = pad_labels[window_slice(center=(y, x), kernel=ker)] # window
                ker_slice = ker_slice.flatten()
                dic = {'Coordinate': (y - pad_h, x - pad_w)}
                for i in range(0, ker_slice.size):
                    dic[str(i)] = ker_slice[i]

                rows.append(dic)

    df = pd.DataFrame(rows)
    rows = []  # Free memory

    df_no_coords = df.drop(['Coordinate'], axis=1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df_no_coords)
    df_no_coords = []  # Free memory
    #kmeans = (GaussianMixture(n_components=4, covariance_type="full", tol=0.001).fit(df_14.reindex(columns=['X', 'Y']))).predict(df_14.reindex(columns=['X', 'Y']))
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    #borders[kmeans.labels_ == 1] = 255
    df_coords = df['Coordinate']
    df = []  # Free memory
    for j in range(0, df_coords.shape[0]):
        if kmeans.labels_[j] == 1 or kmeans.labels_[j] == 2:
            borders[df_coords.get_value(j, 'Coordinate')[0]][df_coords.get_value(j, 'Coordinate')[1]] = 255
    return borders

def gen_phase_mask(restored, orig_img, despeckle_size=0, filter_size=0, file_name="gen_phase_mask", debug=False):
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
    filtered = threshold
    if debug:
        dbgImgs += [(filtered, 'step 2 - pre-filter')]

    # step 3 - dilate
    dilated = cv2.dilate(filtered,kernel,iterations=1)
    #dilated = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        dbgImgs += [(dilated, 'step 3 - dilate')]

    # step 4 - filter far/dead cells
    #filtered = filter_far_cells(dilated, filter_size, debug)
    filtered = dilated

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

def seg_aux_chan(img, frame_id, channel):
    thresh_perc = np.percentile(img, 99.9)
    tmp, thresh = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    thresh_copy = np.copy(thresh)
    thresh = ndimage.median_filter(thresh, size=(3, 3))
    if img_utils.is_noisy(thresh):
        tmp, thresh = cv2.threshold(np.uint8(img), thresh_perc, 255, cv2.THRESH_BINARY)
    thresh = pre_filter_far_cells(thresh, 2)

    cv2.imwrite("dbg\\L136\\A2\\4\\aux_seg\\" + str(frame_id) + "_" + channel + "_copy" + ".tif", thresh_copy)

    cv2.imwrite("dbg\\L136\\A2\\4\\aux_seg\\" + str(frame_id) + "_" + channel + ".tif", thresh)

    return thresh

def get_connected_components(img, grayscale=True, debug=True):

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    img = np.uint8(img)

    connectivity = 4
    labels = cv2.connectedComponentsWithStats(img, connectivity=connectivity)

    return labels

def filter_far_cells(thresh, dev_thresh=1, debug = True):
    return ndimage.median_filter(thresh, size=(3, 3)) # TODO test

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
        print('STD filter: filtered {} out of {} segmented particles. {} cells remain.'.format(filtered_components, init_components, init_components - filtered_components))

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
        print('Min Area filter: filtered {} out of {} segmented particles. {} cells remain.'.format(filtered_components, init_components, init_components - filtered_components))

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


def calc_borders(restored_img, img, despeckle_size, ker):
    #normailize image to unit8 range: 0-255
    restored = cv2.normalize(restored_img, None, 0, 255, cv2.NORM_MINMAX)

    restored = np.uint8(restored)

    # step 1 - threshold
    tmp, threshold = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY)

    # step 2 - filter far/dead cells
    filtered = pre_filter_far_cells(threshold, despeckle_size)

    #filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, (3,3), iterations=1)

    filtered = np.uint8(filtered)

    connectivity = 4
    labels = cv2.connectedComponentsWithStats(filtered, connectivity=connectivity)

    borders = find_borders(restored, ker)

    io_utils.save_img(borders, "dbg\\borders.png")  # TODO Remove

    return borders


def calc_borders_naive(restored_img, img, despeckle_size, ker):
    #normailize image to unit8 range: 0-255
    restored = cv2.normalize(restored_img, None, 0, 255, cv2.NORM_MINMAX)

    restored = np.uint8(restored)

    # step 1 - threshold
    tmp, threshold = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY)

    # step 2 - filter far/dead cells
    filtered = pre_filter_far_cells(threshold, despeckle_size)

    #filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, (3,3), iterations=1)

    filtered = np.uint8(filtered)

    connectivity = 4
    labels = cv2.connectedComponentsWithStats(filtered, connectivity=connectivity)

    borders = find_borders_naive(labels, ker)

    io_utils.save_img(borders, "dbg\\borders_naive.png")  # TODO Remove

    return borders



def fix_segmentation(orig, sub):
    sub = np.uint8(sub)
    tmp_img = np.copy(sub)
    orig_connectivity = 8
    connectivity = 8
    orig_labels = cv2.connectedComponentsWithStats(orig, connectivity=orig_connectivity)
    num_cells_orig = orig_labels[0]
    label_img_orig = orig_labels[1]
    stats = orig_labels[2]

    for cell_id in range(1, num_cells_orig):
        num_cells_sub = cv2.connectedComponentsWithStats(sub, connectivity=connectivity)[0]
        last_img = np.copy(tmp_img)
        tmp_img[label_img_orig == cell_id] = 255
        if cv2.connectedComponentsWithStats(tmp_img, connectivity=connectivity)[0] < num_cells_sub:
            tmp_img = np.copy(last_img)
    return last_img



def seg_phase(img, opt_params=0, ker_params=0, despeckle_size=1, dev_thresh=0, file_name=0, debug=False):
    # First channel with small kernel radius
    additive_zero = ps.prec_sparse(img, opt_params, ker_params, debug)[:, :, 0]
    ker_params_first = KerParams(ring_rad=ker_params.ring_rad, ring_wid=ker_params.ring_wid, ker_rad=ker_params.ker_rad + 1, zetap=ker_params.zetap, dict_size=ker_params.dict_size)
    opt_params_first = OptParams(smooth_weight=opt_params.smooth_weight, spars_weight=opt_params.spars_weight, sel_basis=opt_params.sel_basis, epsilon=opt_params.epsilon, gamma=opt_params.gamma, img_scale=opt_params.img_scale, max_itr=opt_params.max_itr, opt_tolr=opt_params.opt_tolr)

    # Second channel with larger kernel radius
    next = ps.prec_sparse(img, opt_params_first, ker_params_first, debug)
    next = next[:, :, 0]

    # Stack both first channels (enhances results)
    additive_zero = cv2.add(additive_zero, next)

    # Create initial mask
    post_proc = gen_phase_mask(additive_zero, img, despeckle_size=3, filter_size=3, file_name=file_name, debug=debug)

    # Calculate borders
    borders = calc_borders(additive_zero, img, 1, ker=(5, 5))
    borders_naive = calc_borders_naive(additive_zero, img, 1, ker=(5, 5))

    additive_zero = []  # Free memory
    next = []  # Free memory

    # Subtract borders from initial mask
    sub_filter_naive = cv2.subtract(post_proc, np.uint8(borders_naive))
    #sub_filter_naive = pre_filter_far_cells(sub_filter_naive, despeckle_size=3) # 1-9 pixel cells are most likely noise
    #sub_filter_naive = filter_far_cells(sub_filter_naive, dev_thresh=1.8)
    sub_filter_naive = fix_segmentation(post_proc, sub_filter_naive)

    # Subtract borders from initial mask
    sub_filter = cv2.subtract(post_proc, np.uint8(borders))
    sub_filter = pre_filter_far_cells(sub_filter, despeckle_size=9) # 1-9 pixel cells are most likely noise
    sub_filter = filter_far_cells(sub_filter, dev_thresh=1.7)

    sub_filter_merged = cv2.bitwise_or(np.uint8(sub_filter), sub_filter_naive)

    # Fix segmentation
    fixed = fix_segmentation(post_proc, sub_filter_merged)

    io_utils.save_img(sub_filter_naive, "dbg\\test_sub_filter.png", uint8=True)  # TODO Remove
    io_utils.save_img(post_proc, "dbg\\test_proc.png", uint8=True)  # TODO Remove
    io_utils.save_img(fixed, "dbg\\test_sub_filter_fixed.png", uint8=True)  # TODO Remove
    io_utils.save_img(sub_filter, "dbg\\test_sub_filter_kmeans.png", uint8=True)  # TODO Remove

    return fixed


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
