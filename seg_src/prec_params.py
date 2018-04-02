"""
Contains parameter classes used in precondition

KerParams represents the kernel parameters.

OptParams represents the optimization parameters.
"""

import numpy as np


class KerParams:
    """
    Represents kernel parameters.

    Parameters
    ----------
    ring_rad : float
        Phase ring outer radius.
    ring_wid : float
        Phase ring width.
    ker_rad : float
        Kernel radius.
    zetap : float
        Amplitude attenuation factors caused by phase ring.
    dict_size : int
        Size of dictionary.
    """
    def __init__(self, ring_rad=4, ring_wid=0.8, ker_rad=2, zetap=0.8, dict_size=20):
        self.ring_rad = ring_rad # Phase ring outer radius
        self.ring_wid = ring_wid # Phase ring width
        self.ker_rad = ker_rad # Kernel radius
        self.zetap = zetap # Amplitude attenuation factors caused by phase ring
        self.dict_size = dict_size # Size of dictionary


class OptParams:
    """
    Represents optimization parameters.

    Parameters
    ----------
    smooth_weight : float
        Spatial smoothness term weight.
    spars_weight : float
        Sparsity term weight.
    sel_basis : int
        Max selected basis.
    epsilon : float
        Part of smoothing term.
    gamma : int
        Part of re-weighting term.
    img_scale : float
        Image scaling factor.
    max_itr : int
        Max number of optimization iterations.
    opt_tolr : float
        Optimization tolerance.
    """
    def __init__(self, smooth_weight=1, spars_weight=0.4, sel_basis=3, epsilon=3, gamma=3, img_scale=1, max_itr=100, opt_tolr=np.finfo(float).eps):
        self.smooth_weight = smooth_weight # Spatial smoothness term weight
        self.spars_weight = spars_weight # Sparsity term weight
        self.sel_basis = sel_basis # Max selected basis
        self.epsilon = epsilon # Part of smoothing term
        self.gamma = gamma # Part of re-weighting term
        self.img_scale = img_scale # Image scaling factor
        self.max_itr = max_itr # Max number of optimization iterations
        self.opt_tolr = opt_tolr # Optimization tolerance
