import cv2 as cv
import numpy as np
import debug_utils as dbg
import img_utils as iu

def generate_mask(raw_threshold, orig_img, debug=True):
    dbgImgs = []
    if debug:
        dbgImgs += [(raw_threshold, 'initial')]

    #normailize image to unit8 range: 0-255
    raw_threshold = cv.normalize(raw_threshold, None, 0, 255, cv.NORM_MINMAX)
    orig_img = cv.normalize(orig_img, None, 0, 255, cv.NORM_MINMAX)
    raw_threshold = np.uint8(raw_threshold)
    orig_img = np.uint8(orig_img)

    kernel = np.ones((3, 3), np.uint8)

    # step 1 - threshold
    tmp, threshold = cv.threshold(raw_threshold, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if debug:
        dbgImgs += [(threshold, 'step 1 - threshold')]

    # step 2 - dialate
    dialated = cv.dilate(threshold,kernel,iterations = 1)

    if debug:
        dbgImgs += [(dialated, 'step 2 - dialate')]

    # step 3 - borders
    borders = cv.Canny(dialated,100,200)

    if debug:
        dbgImgs += [(borders, 'step 4 - borders')]

    # step 4 - borders
    compare = np.copy(orig_img)
    compare[np.nonzero(borders == 255)] = 255

    if debug:
        dbgImgs += [(compare, 'step 4 - compare')]


    if debug:
        dbg.save_debug_fig(dbgImgs, 'generate_mask', zoom=5)

