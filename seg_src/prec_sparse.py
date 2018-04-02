import sys

import prec_utils
import img_utils
import numpy as np
import cv2
import matplotlib.pyplot as plt

from debug_utils import save_debug_fig


def prec_sparse(img, opt_params, ker_params, debug):
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

    # Resize image if required
    if opt_params.img_scale != 1:
        img = img_utils.resize_img(img, opt_params.img_scale)

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
        print("{}th basis generation\n", sel_ind)
        kernel = prec_utils.get_kernel(ker_params, sel_basis[sel_ind] / (2 * dict_size * np.math.pi), 0)

        # Calculate basis for kernel
        basis = prec_utils.calc_basis(kernel, img_dim[0], img_dim[1])

        # Calculate coefficient
        print("Calculate coefficient of the {}th basis\n", sel_ind)
        resd_img[:,:, sel_ind] = np.reshape(prec_utils.phase_seg(basis, rimg, opt_params, debug), (img_dim[0], img_dim[1]), order='F')

        # Update residual error
        print("Residual error update\n")
        rimg = rimg - np.reshape(basis * resd_img[(1 + (sel_ind - 1) * rimg.size()) : (sel_ind * rimg.size())].getH(),img_dim[0], img_dim[1])

        # Normalize
        img_proc[:, :, sel_ind] = img_utils.normalize(resd_img[:, :, sel_ind])

        if debug:
            # Display residual and restored images
            imgs = [(rimg, 'Residual image:'), (img_proc, 'Restored image:')]
            save_debug_fig(imgs, 'prec_sparse.png')

        if np.norm(resd_img[:, :, sel_ind]) / np.norm(resd_img[:, :, 1]) < 0.01:
            break

    ndep = np.size(img_proc, 3)
    img_proc[:, :, ndep + 1: num_basis] = 0

    return img_proc
