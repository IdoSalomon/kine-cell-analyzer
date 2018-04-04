import numpy as np
import cv2
import prec_sparse as ps
import process as pr
import img_utils as iu
from prec_params import KerParams, OptParams
import random


def main():
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=1, max_itr=100, opt_tolr=np.finfo(float).eps)
    img = iu.load_img("images\\small.png", True)

    res_img = ps.prec_sparse(img, opt_params, ker_params, False)
    red_channel = res_img[:, :, 0 ]
    pr.generate_mask(red_channel, img)
    #pr.colorConnectedComponents(pr.generate_mask(red_channel, img))


if __name__== "__main__":
    main()