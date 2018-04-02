import numpy as np

import prec_sparse as pr
from prec_params import KerParams, OptParams


def main():
    print("hello")

    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=3, epsilon=3, gamma=3, img_scale=2, max_itr=100, opt_tolr=np.finfo(float).eps)
    img = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    pr.prec_sparse(img, opt_params, ker_params, False)

if __name__== "__main__":
    main()