import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import debug_utils as dbg


def bg_removal(img, debug=False):
    """
    Removes background from image

    Parameters
    ----------
    img : ndarray
        2D array that represents image
    debug : bool
        Boolean that dictates if function is in debug mode

    Returns
    -------
    flattened : ndarray
        2D array that represents image w/o background
    """

    # copy source image
    bg = np.copy(img)
    
    print('src image:')
    print(bg) # Debug

    # get image dimensions
    rows_no = np.size(bg, 0)
    cols_no = np.size(bg, 1)
    N = cols_no * rows_no

    # create grids
    xx, yy = np.meshgrid(np.arange(1., cols_no + 1), np.arange(1., rows_no + 1))

    print(xx) # Debug
    print(yy) # Debug

    # turn grids 2D arrays into vectors (F - column major)
    xx = xx.flatten(order='F')
    yy = yy.flatten(order='F')

    print(xx) # Debug
    print(yy) # Debug

    X = np.array([np.ones(N), xx, yy, (xx ** 2), (xx * yy), (yy ** 2)]).T
    print('X=')
    print(X)
    # solve linear system
    p = np.array(np.linalg.lstsq(X, img.flatten(order='F'), rcond=None)[0])
    print('p=')
    print(p) # Debug

    # np.reshape()
    #
    print('im(:)=')
    print(img.flatten(order='F')) # Debug
    print('X*p=')
    print(np.dot(X, p)) # Debug


    res = img.flatten(order='F') - np.dot(X, p)
    print("im(:)-X*p=")
    print(res)
    flattened = np.reshape(res, np.array([rows_no, cols_no]), order='F')
    print("flattened result =")
    print(flattened) # Debug

    print('bg =')
    bg = np.reshape(np.dot(X, p), (rows_no, cols_no), order='F')

    print(bg) # Debug

    if debug:
        imgs = [(img, 'original image:'), (flattened, 'flattened image:'), (bg, 'background image:')]
        dbg.save_debug_fig(imgs, 'bgRemoval.png')
        cv2.imwrite("dbg\\bg_removal_flattened.png", flattened)

    return flattened


def normalize(img):
    """
    Normalize the image elements to [0, 1]

    Parameters
    ----------
    img : ndarray
        2D array that represents image

    Returns
    -------
        2D array of normalized image
    """

    min_val = min(img[:])
    max_val = max(img[:])

    if min_val == max_val:
        return np.zeros(np.size(img));

    norm = (img - min_val) / (max_val - min_val)

    return np.sqrt((2 * norm) - np.power(norm, 2))


def get_dim(img):
    """
    Returns image dimensions

    Parameters
    ----------
    img : ndarray
        2D array that represents image

    Returns
    -------
    (image rows, image columns)
    """

    try:
        img_dim = (img.shape[0], img.shape[1])
    except IndexError:
        img_dim = (1, img.shape[0])
    return img_dim


def resize_img(img, img_scale):
    """

    Parameters
    ----------
    img : ndarray
        2D array that represents image

    img_scale : int
        Scale factor

    Returns
    -------
    img_resize : ndarray
        2D array of resized image
    """

    img_dim = get_dim(img)

    if img_scale < 1:
        # Shrink image
        img_resize=cv2.resize(img, (round(img_dim[1] * img_scale), round(img_dim[0] * img_scale)), interpolation=cv2.INTER_AREA) # Expects opencv compatible array
        return img_resize
    if img_scale > 1:
        # Enlarge image
        img_resize=cv2.resize(img, (img_dim[1] * img_scale, img_dim[0] * img_scale), interpolation=cv2.INTER_CUBIC) # Expects opencv compatible array
        #img_resize = np.array(img_resize) # convert image to ndarray
        return img_resize

    return img

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

if __name__ == "__main__":
    img = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    bg_removal(img, True)
