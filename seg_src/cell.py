class Cell:
    """
    Represents cell.

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
    def __init__(self, frame_label, global_label=1, area=0, centroid=(0,0), perimeter=0, circularity=0, pixels=0, pixel_values={}):
        self.frame_label = frame_label
        self.global_label = global_label
        self.area = area
        self.centroid = centroid
        self.perimeter = perimeter
        self.circularity = circularity
        self.pixels = pixels
        self.pixel_values = pixel_values

