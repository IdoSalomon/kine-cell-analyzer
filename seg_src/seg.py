import numpy as np
import prec_sparse as ps
import process as pr
import img_utils as iu
from prec_params import KerParams, OptParams
import img_utils
import random
import prec_params
import os
import cv2


def get_gradient(img):
    # Calculate the x and y gradients using Sobel operator
    grad_x = np.float32(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    grad_y = np.float32(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    return grad

def align_img(img_to_align, img_ref):
    img_to_align = iu.load_img("images\\L136\\A2\\4\\L136_phase_A2_4_2018y02m12d_10h30m.tif", 0.5, True, False)
    img_ref = iu.load_img("images\\L136\\A2\\4\\L136_phase_A2_4_2018y02m12d_10h45m.tif", 0.5, True, False)
    img_aligned = img_to_align

    # Find the width and height of the color image
    size = img_ref.shape
    height = size[0]
    width = size[1]

    # Define motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Set the warp matrix to identity.
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500000000, 0.00001)

    # Warp the blue and green channels to the red channel
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(img_ref), get_gradient(img_to_align),
                                             warp_matrix, warp_mode, criteria)
    #tmp_align = np.uint8(img_to_align)
    #tmp_ref = np.uint8(img_ref)
    #warp_matrix = cv2.estimateRigidTransform(tmp_align, tmp_ref, True)
    # Use Affine warp when the transformation is not a Homography
    img_aligned = cv2.warpAffine(img_to_align, warp_matrix, (width, height),
                                      flags=cv2.WARP_INVERSE_MAP)

    # Show final output
    cv2.imwrite("dbg\\ColorImage.png", cv2.normalize(img_ref, None, 0, 255, cv2.NORM_MINMAX))
    cv2.waitKey(0)

    cv2.imwrite("dbg\\ColorAlignedImage.png", cv2.normalize(img_aligned, None, 0, 255, cv2.NORM_MINMAX))
    cv2.waitKey(0)

def main():
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=1, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5, max_itr=100, opt_tolr=np.finfo(float).eps)
    img = iu.load_img("images\\seq_nec\\Scene1Interval001_PHASE.png", 0.5, True, False)

    align_img(img, img)
    #res_img = pr.seg_phase(img, despeckle_size=5, dev_thresh=1, ker_params=ker_params, opt_params=opt_params, file_name="test", debug=True)

    #pr.gen_phase_mask(red_channel, img)

if __name__== "__main__":
    main()
    # analyzeDir('images')



