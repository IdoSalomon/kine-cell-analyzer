import os
import cv2
import numpy as np

img_extensions = [".tif", ".png", ".jpg"] # Valid file extensions to load

def extract_name(title):
    """
    Extracts image name from file title.

    Assumes title format such that name is separated from rest of the title by '_'

    Parameters
    ----------
    title : str
        file name

    Returns
    -------
        image name on success, "" otherwise
    """
    if title:
        name = title.split('_') # removes channel and image type information
        if len(name) > 0:
            return name[0]
    return ""


def extract_num(title):
    """
    Extracts image (interval) number from file title.

    Assumes title format such that last number in title is the positive image number.

    Parameters
    ----------
    title : str
        file name

    Returns
    -------
        Image number string on success, "" otherwise
    """
    if title:
        numbers = [int(s) for s in title if s.isdigit()]
        if len(numbers) > 0:
            return "".join(str(numbers[-3:]))
    return ""


def load_paths(dir):
    paths = {}
    for r, d, f in os.walk(dir):
        for file in f:
            filename, file_extension = os.path.splitext(file)
            if file_extension not in img_extensions:
                break
            name = extract_name(file)
            if name:
                if name in paths:
                    paths[name].append("\\".join([dir,file]))
                else:
                    path_lst = ["\\".join([dir,file])]
                    paths[name] = path_lst
    return paths

def save_img(img, path, grayscale=True): #TODO: change to 16 bits
    img = np.uint8(img)

    cv2.imwrite(path, img)