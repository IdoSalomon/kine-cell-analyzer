import matplotlib.pyplot as plt
import math
import os

DBG_DIR = 'dbg'

def save_debug_fig(images, fig_name, ncols=2):
    nrows = math.ceil(len(images) / float(ncols))

    fig = plt.figure(figsize=(ncols*3, nrows*3))

    for i in range(1, len(images) + 1):
        print(str(i))
        fig.add_subplot(nrows, ncols, i)
        plt.imshow(images[i-1][0], cmap=plt.cm.gray)
        plt.title(images[i-1][1])
    plt.tight_layout()
    plt.show()

    if not os.path.exists(DBG_DIR):
        os.makedirs(DBG_DIR)

    fig.savefig(DBG_DIR + '/' + fig_name)
