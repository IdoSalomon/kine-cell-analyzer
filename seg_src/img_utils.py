import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg


def bg_removal(img, debug=False):
    '''
    :param img:
    :param debug:
    :return:
    '''

    # copy source image
    bg = np.copy(img)
    print(bg)

    # get image dimensions
    rows_no = np.size(bg, 0)
    cols_no = np.size(bg, 1)
    N = cols_no * rows_no

    # create grids
    xx, yy = np.meshgrid(np.arange(1., cols_no + 1), np.arange(1., rows_no + 1))

    print(xx)
    print(yy)

    # turn grids 2D arrays into vectors (F - column major)
    xx = xx.flatten(order='F')
    yy = yy.flatten(order='F')

    print(xx)
    print(yy)

    X = np.array([np.ones(N), xx, yy, (xx ** 2), (xx * yy), (yy ** 2)]).T

    # solve linear system
    p = np.linalg.lstsq(X, img.flatten(order='F'), rcond=None)[0]
    print(p[0])

    # np.reshape()
    #
    print('last')
    print(img.flatten(order='F'))
    print(np.dot(X, p))
    res = np.subtract(img.flatten(order='F'), np.dot(X, p))
    flattened = np.reshape(res, np.array([rows_no, cols_no]))
    bg = np.reshape(np.dot(X, p), np.array([rows_no, cols_no]))

    print(img)
    print(bg)
    fig = plt.figure()
    fig.add_subplot(3, 3, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    
    plt.title('original image:')
    fig.add_subplot(3, 3, 2)
    plt.imshow(flattened, cmap=plt.cm.gray)
    plt.title('flattened image:')
    fig.add_subplot(3, 3, 3)
    plt.imshow(bg, cmap=plt.cm.gray)
    plt.title('background image:')
    fig.savefig('testFig.png')

    return flattened

def normalize(img):
    """
    Normalize the image [0, 1]

    Parameters
    ----------
    img : ndarray
        3D array that represents image

    Returns
    -------
    Normalized image

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
        3D array that represents image

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
    if img_scale < 1:
        # Enlarge image
        img_resize = cv2.resize(img,(img_scale, img_scale), cv2.INTER_AREA) # TODO redo cv
        img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    elif img_scale > 1:
        # Shrink image
        img_resize = cv2.resize(img,(img_scale, img_scale), cv2.INTER_CUBIC) # TODO redo cv
        img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    img = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    bg_removal(img)
