import sys

import scipy

import prec_utils
import img_utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import misc

from debug_utils import save_debug_fig


def prec_sparse(img, opt_params, ker_params, debug=True):
    """
    Creates reconditioned image

    Parameters
    ----------
    img : ndarray
        2D array that represents image
    opt_params : OptParams
        optimization parameters
    ker_params : KerParams
        kernel parameters
    debug : bool
        is in debug mode
    Returns
    -------
    img_proc : ndarray
        2D array of preconditioned image
    """

    # Validate args
    if prec_utils.validate_params(ker_params, opt_params) == False:
        print("Invalid arguments. Exiting...")
        sys.exit()

    # Remove image background
    print("Removing background\n")
    img = img_utils.bg_removal(img, debug)

    # Sparse representation
    num_basis = opt_params.sel_basis # total num of basis
    dict_size = ker_params.dict_size # dictionary size
    sel_basis = np.zeros(num_basis) # init basis array
    rimg = img.copy() # initial residual image
    img_dim = img_utils.get_dim(img) # image dimensions
    resd_img = np.zeros((img_dim[0], img_dim[1], num_basis)) # 3d array to hold residual images
    img_proc = np.zeros((img_dim[0], img_dim[1], num_basis)) # 3d array to hold proc image

    # Iterate over all basis
    for sel_ind in range(num_basis):
        # Select the sel_ind basis
        print("Select the best basis\n")
        sel_basis[sel_ind] = prec_utils.basis_select(rimg, ker_params, dict_size, debug)

        # Get kernel for basis
        print("{} basis generation\n".format(str(sel_ind)))
        kernel = prec_utils.get_kernel(ker_params, ((sel_basis[sel_ind] + 1) / dict_size) * 2 * np.math.pi, 0)

        # Calculate basis for kernel
        basis = prec_utils.calc_basis(kernel, img_dim[0], img_dim[1])

        # Calculate coefficient
        print("Calculate coefficient of the {}th basis\n".format(str(sel_ind)))
        resd_img[:,:, sel_ind] = np.reshape(prec_utils.phase_seg(basis, rimg, opt_params, debug), (img_dim[0], img_dim[1]), order='F')

        # Update residual error
        print("Residual error update\n")
        rimg = rimg - np.reshape(basis.dot(resd_img[:,:,sel_ind].flatten('F').conj().T), (img_dim[0], img_dim[1]), order='F')

        # Normalize
        img_proc[:, :, sel_ind] = img_utils.normalize(resd_img[:, :, sel_ind])
        img_copy = np.copy(img_proc)

        if debug:
            # Display residual and restored images
            imgs = [(rimg, 'Residual image:'), (img_proc[:, :, sel_ind], 'Restored image:')]
            save_debug_fig(imgs, 'prec_sparse.png')
            # Save channel image
            rgb = cv2.normalize(img_copy[:, :, sel_ind], None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite("dbg\\restored_image_" + str(sel_ind) + ".png", rgb)

        if np.linalg.norm(resd_img[:, :, sel_ind]) / np.linalg.norm(resd_img[:, :, 0]) < 0.01:
            break

    ndep = img_proc.shape[2]
    img_proc[:, :, ndep: num_basis] = 0
    #img_proc[:,:,2] = 0 # TODO REMOVE
    # Save final BGR image
    bgr = cv2.normalize(img_proc, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("dbg\\restored_image.png", bgr)

    return img_proc
