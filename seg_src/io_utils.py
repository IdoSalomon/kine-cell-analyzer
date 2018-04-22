import os
import cv2
import numpy as np

img_extensions = [".tif", ".png", ".jpg"] # Valid file extensions to load

def extract_name(title):
    """
    Extracts image name from file title.

    Assumes title format such that channel name is separated from rest of the title by '_'

    Parameters
    ----------
    title : str
        file name

    Returns
    -------
        image name on success, "" otherwise
    """
    if title:
        filename, file_extension = os.path.splitext(title)
        name = filename.split('_') # removes channel and image type information
        if len(name) > 0:
            return name[0]
    return ""


def extract_num(title):
    """
    Extracts image (interval) number from file title.

    Assumes title format such that last number in title is the positive image number comprised of 3 digits.

    Parameters
    ----------
    title : str
        file name

    Returns
    -------
        Image number string on success, -1 otherwise
    """
    if title:
        numbers = [int(s) for s in title if s.isdigit()]
        if len(numbers) > 0:
            return int("".join([str(dig) for dig in numbers[-3:]]))
    return -1


def load_paths(dir):
    paths = {}
    for f in os.listdir(dir):
        filename, file_extension = os.path.splitext(f)
        if file_extension in img_extensions:
            name = extract_name(f)
            if name:
                if name in paths:
                    paths[name].append("\\".join([dir, f]))
                else:
                    path_lst = ["\\".join([dir, f])]
                    paths[name] = path_lst
    return paths

def save_img(img, path, grayscale=True): #TODO: change to 16 bits
    img = np.uint8(img)

    cv2.imwrite(path, img)