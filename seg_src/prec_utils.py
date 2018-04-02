import prec_params as param
import img_utils as imtil
import debug_utils as dbg
import numpy as np
import scipy as sci
import math


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


def calc_basis(kernel, nrows, ncols):

    diameter = np.size(kernel, 1)
    radius = round((diameter - 1) / 2.0)
    kernel = kernel.flatten(order='F')
    N = nrows * ncols

    # build sparse matrix H
    logic_arr = (abs(kernel) > 0.01)
    logic_arr = np.array([int(x) for x in logic_arr])

    inds = np.reshape(np.arange(1, N+1), (nrows, ncols), order='F')

    inds_pad = np.pad(inds, np.array([radius, radius]), mode='symmetric')

    rows_inds = np.tile(np.arange(1, N+1), (sum(logic_arr), 1))
    col_inds = im2col_sliding_strided(inds_pad.T, (diameter, diameter))
    print(col_inds)

    logic_arr = logic_arr[:, None]
    col_inds = np.array([col_inds[i][j]np.tile(logic_arr, (1,N))])

    print(col_inds)
    print("CALLED CALC_BASIS")

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


if __name__ == "__main__":
    kernel = np.array([[2,1,0], [0, 1, 0], [0, 0, 1]])
    calc_basis(kernel, 3, 3)
