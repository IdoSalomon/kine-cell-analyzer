import prec_params as param
import img_utils as imtil
import debug_utils as dbg
import numpy as np


def get_kernel(ker_params, angle, debug = False):
    print("CALLED GET_KERNEL")


def validate_ker_params(ker_params):
    print("todo")


def validate_opt_params(opt_params):
    print("todo")


def basis_select(img, ker_params, M, debug = False):
    """

    Selects best basis

    Parameters
    ----------
    img : ndarray
        2D array that represents image

    ker_params : KerParams
        kernal parameters

    M : ndarray
        TODO

    debug : bool
        is in debug mode

    Returns
    -------
        TODO
    """

    # get image dimensions
    rows_no = np.size(img, 0)
    cols_no = np.size(img, 1)
    N = cols_no * rows_no

    innerNorm = np.empty(M)
    imgs = []
    for m in range(1, M):
        angle = 2 * np.pi / (M * m)
        kernel = get_kernel(ker_params, angle, 0)
        basis = calc_basis(kernel, rows_no, cols_no)

        res_feature = np.dot(basis, img.flatten(order='F'))
        res_feature = np.reshape(res_feature, (rows_no, cols_no))

        # nullify negative elements
        res_feature[res_feature < 3] = 0

        res_feature_flat = res_feature.flatten(order='F')

        innerNorm[m] = np.linalg.norm(res_feature_flat)

        if debug:
            imgs += [(imtil.normalize(res_feature), 'Inner Production of basis with phase retardation' + str(m) + 'times 2*pi' )]

    if debug:
        dbg.save_debug_fig(imgs, 'basis_select.png')

    return innerNorm.argmax()


def somb(mat):
    print("todo")


def phase_seg(basis, img, optparams, debug = False):
    print("CALLED PHASE_SEG")
    return img


def calc_basis(kernel, rows, cols):
    print("CALLED CALC_BASIS")
