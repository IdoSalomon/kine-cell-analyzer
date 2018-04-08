import numpy as np
import prec_sparse as ps
import process as pr
import img_utils as iu
from prec_params import KerParams, OptParams
import img_utils
import random
import prec_params
import os



def main():
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=1, epsilon=3, gamma=3, img_scale=0.5, max_itr=100, opt_tolr=np.finfo(float).eps)
    img = iu.load_img("images\\Scene1Interval077_TRANS.tif", 0.5, True, False)

    res_img = ps.prec_sparse(img, opt_params, ker_params, False)
    red_channel = res_img[:, :, 0]

    pr.gen_phase_mask(red_channel, img)

# todo move this
def create_mask_dir(dir, ker_params, opt_params):
    for r, d, f in os.walk(dir):
        for file in f:
            # Load all phase images
            if 'PHASE' in file and ('.png' in file or '.tif' in file):
                img = iu.load_img(dir + '/' + file, opt_params.img_scale, True)
                res_img = ps.prec_sparse(img, opt_params, ker_params, False)
                red_channel = res_img[:, :, 0]
                # color components
                pr.get_connected_components(pr.gen_phase_mask(red_channel, False))


if __name__== "__main__":
    main()
    # analyzeDir('images')



