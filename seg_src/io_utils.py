import os
import cv2
import numpy as np
import misc_params as par

img_extensions = [".tif", ".png", ".jpg"] # Valid file extensions to load
date_id = {} # Maps dates to image ID

img_index = 0

def extract_name(title, format=par.TitleFormat.DATE):
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

        if format == par.TitleFormat.SCENE:
            name = filename.split('_') # removes channel and image type information
            if len(name) > 0:
                return name[0]

        elif format == par.TitleFormat.DATE:
            splt = filename.split('_')
            del splt[1]
            del splt[-2:]
            splt.append('%03d' % extract_num(title, format)) # TODO reconsider removing id from name
            name = ("_".join(splt))
            if len(name) > 0:
                return name
        elif format == par.TitleFormat.TRACK:
            return filename

        """elif format == par.TitleFormat.DATE:
            splt = filename.split('_')
            name = ("".join(splt[0:3]))
            if len(name) > 0:
                return name[0]"""

    return ""


def extract_date(title, format=par.TitleFormat.DATE):
    """
    Extracts image date from file title.

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

        if format == par.TitleFormat.DATE:
            splt = filename.split('_')
            date = ("_".join(splt[-2:]))
            if len(date) > 0:
                return date

        """elif format == par.TitleFormat.DATE:
            splt = filename.split('_')
            name = ("".join(splt[0:3]))
            if len(name) > 0:
                return name[0]"""

    return ""


def extract_num(title, format=par.TitleFormat.DATE):
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
        if format == par.TitleFormat.SCENE:
            digits = [int(s) for s in title if s.isdigit()]
            if len(digits) > 0:
                return int("".join([str(dig) for dig in digits[-3:]]))

        elif format == par.TitleFormat.DATE:
            index = -1
            date = extract_date(title, format)

            global img_index

            if date not in date_id:
                img_index += 1
                index = img_index
                date_id[date] = index
            else:
                index = date_id[date]

            return index
        elif format == par.TitleFormat.TRACK:
            return 1

        """elif format == par.TitleFormat.DATE:
            filename, file_extension = os.path.splitext(title)
            splt = filename.split('_')
            digits = ("".join(splt[-2:]))
            digits = [int(s) for s in digits if s.isdigit()]
            if len(digits) > 0:
                return digits"""
    return -1


def load_paths(dir, format=par.TitleFormat.DATE):
    paths = {}
    for f in os.listdir(dir):
        filename, file_extension = os.path.splitext(f)
        if file_extension in img_extensions:
            name = extract_name(f, format=format)
            if name:
                if "trk-" in name:
                    name = filename
                if name in paths:
                    paths[name].append("/".join([dir, f]))
                else:
                    path_lst = ["/".join([dir, f])]
                    paths[name] = path_lst
    return paths

def save_img(img, path, uint8=False): #TODO: change to 16 bits
    if uint8:
        img = np.uint8(img)
    else:
        img = np.uint16(img)

    cv2.imwrite(path, img)