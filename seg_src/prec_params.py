"""
KerParams represents the kernel parameters.
OptParams represents the optimization parameters.
"""

import numpy as np

# Represents kernel parameters -
#   Phase ring outer radius.
#   Phase ring width.
#   Kernel radius.
#   Amplitude attenuation factors caused by phase ring.
#   Size of dictionary.
class KerParams:
    def __init__(self, ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20):
        self.ring_rad = ring_rad # Phase ring outer radius
        self.ring_wid = ring_wid # Phase ring width
        self.ker_rad = ker_rad # Kernel radius
        self.zetap = zetap # Amplitude attenuation factors caused by phase ring
        self.dict_size = dict_size # Size of dictionary


# Represents optimization parameters -
#   Spatial smoothness term weight
#   Sparsity term weight
#   Max selected basis
#   Epsilon (part of smoothing term)
#   Gamma (part of re-weighting term)
#   Image scaling factor
#   Max number of optimization iterations
#   Optimization tolerance
class OptParams:
    def __init__(self, smooth_weight=1, spars_weight=0.4, sel_basis=3, epsilon=3, gamma=3, img_scale=1, max_itr=100, opt_tolr=np.finfo(float).eps):
        self.smooth_weight = smooth_weight # Spatial smoothness term weight
        self.spars_weight = spars_weight # Sparsity term weight
        self.sel_basis = sel_basis # Max selected basis
        self.epsilon = epsilon # Part of smoothing term
        self.gamma = gamma # Part of re-weighting term
        self.img_scale = img_scale # Image scaling factor
        self.max_itr = max_itr # Max number of optimization iterations
        self.opt_tolr = opt_tolr # Optimization tolerance
