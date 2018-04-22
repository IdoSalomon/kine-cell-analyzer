"""
Contains image frame class

"""

import numpy as np
from enum import Enum

class FrameState(Enum):
    """ Represents frame state """
    INIT = 0
    PROCESSED = 1




class Frame:
    """
    Represents image frame.

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
    def __init__(self, frame_state=FrameState.INIT, title="", id=0, name="", cells={}, images={}, masks={}, con_comps=[], tracked_img={}):
        self.frame_state = frame_state
        self.title = title
        self.name = name
        self.id = id
        self.cells = cells
        self.images = images
        self.masks = masks
        self.con_comps = con_comps
        self.tracked_img = tracked_img

    def add_image(self, image):
        self.images.append(image)

    def add_mask(self, mask):
        self.masks.append(mask)
