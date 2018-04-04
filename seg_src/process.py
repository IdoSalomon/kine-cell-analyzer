import cv2
import numpy as np
import debug_utils as dbg
import img_utils as iu

def generate_mask(raw_threshold, orig_img, debug=True):
    dbgImgs = []
    if debug:
        dbgImgs += [(raw_threshold, 'initial')]

    #normailize image to unit8 range: 0-255
    raw_threshold = cv2.normalize(raw_threshold, None, 0, 255, cv2.NORM_MINMAX)
    orig_img = cv2.normalize(orig_img, None, 0, 255, cv2.NORM_MINMAX)
    raw_threshold = np.uint8(raw_threshold)
    orig_img = np.uint8(orig_img)

    kernel = np.ones((3, 3), np.uint8)

    # step 1 - threshold
    tmp, threshold = cv2.threshold(raw_threshold, 0, 255, cv2.THRESH_BINARY)

    if debug:
        dbgImgs += [(threshold, 'step 1 - threshold')]

    # step 2 - dialate
    dialated = cv2.dilate(threshold,kernel,iterations = 1)

    if debug:
        dbgImgs += [(dialated, 'step 2 - dialate')]

    # step 3 - borders
    borders = cv2.Canny(dialated,100,200)

    if debug:
        dbgImgs += [(borders, 'step 4 - borders')]

    # step 4 - borders
    compare = np.copy(orig_img)
    compare[np.nonzero(borders == 255)] = 255

    if debug:
        dbgImgs += [(compare, 'step 4 - compare')]
        dbgImgs += [(orig_img, 'original image')]

    if debug:
        dbg.save_debug_fig(dbgImgs, 'generate_mask', zoom=5)


def colorConnectedComponents(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    img = np.uint8(img)

    ret, labels = cv2.connectedComponents(img, connectivity=4)

    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()