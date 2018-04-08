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
    def __init__(self, frame_state=FrameState.INIT, title="", num="", name="", cells={}, images={}, masks={}, con_comps=[], tracked_mask=[]):
        self.frame_state = frame_state
        self.title = title
        self.name = name
        self.num = num
        self.cells = cells
        self._images = images
        self._masks = masks
        self.con_comps = con_comps
        self.tracked_mask = tracked_mask

    def add_image(self, image):
        self.images.append(image)

    def add_mask(self, mask):
        self.masks.append(mask)


    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def images(self, masks):
        self._masks = masks

