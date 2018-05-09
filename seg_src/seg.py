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
    ker_params = KerParams(ring_rad=4, ring_wid=0.8, ker_rad=1, zetap=0.8, dict_size=20)
    opt_params = OptParams(smooth_weight=1, spars_weight=0.4, sel_basis=3, epsilon=3, gamma=3, img_scale=0.5, max_itr=100, opt_tolr=np.finfo(float).eps)
    img = iu.load_img("images\\seq_nec\\Scene1Interval001_PHASE.png", 0.5, True, False)

    res_img = pr.seg_phase(img, despeckle_size=5, dev_thresh=1, ker_params=ker_params, opt_params=opt_params, file_name="test", debug=True)

    #pr.gen_phase_mask(red_channel, img)

if __name__== "__main__":
    main()
    # analyzeDir('images')



