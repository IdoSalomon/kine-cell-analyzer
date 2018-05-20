import math
import os

import matplotlib.pyplot as plt

DBG_DIR = 'dbg'

def save_debug_fig(images, fig_name, ncols=2, zoom = 3):
    nrows = math.ceil(len(images) / float(ncols))

    fig = plt.figure(figsize=(ncols*zoom, nrows*zoom))

    for i in range(1, len(images) + 1):
        print(str(i))
        fig.add_subplot(nrows, ncols, i)
        plt.imshow(images[i-1][0], 'gray')
        plt.title(images[i-1][1])
    plt.tight_layout()
    plt.show()

    if not os.path.exists(DBG_DIR):
        os.makedirs(DBG_DIR)

    fig.savefig(DBG_DIR + '/' + fig_name)


def plot_kinematics(cell_frames, cell_trans, frames_No, red_chan='PI', green_chan='fitc', all_frames=True):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(40, 8))  # 400 for full size

    if all_frames:
        context = cell_frames
    else:
        context = cell_trans

    # labels

    cell_label = [i for i in sorted(context) if i < 1000]

    first_appeared_filter = [min(cell_frames[i]) - 1 for i in cell_label]

    red = [0 if i not in cell_trans or red_chan not in cell_trans[i] else 1 + frames_No - cell_trans[i][red_chan] for i
           in cell_label]
    green = [0 if i not in cell_trans or green_chan not in cell_trans[i] else 1 + frames_No
                                                                              - cell_trans[i][green_chan] for i in
             cell_label]

    # red before green WA
    green_if_red_before = [0 if i not in cell_trans
                                or red_chan not in cell_trans[i]
                                or green_chan not in cell_trans[i]
                                or cell_trans[i][green_chan] <= cell_trans[i][red_chan]
                           else 1 + frames_No - cell_trans[i][green_chan]
                           for i in cell_label]

    red_bottom = [0 if i not in cell_trans or red_chan not in cell_trans[i] else cell_trans[i][red_chan] - 1 for i in
                  cell_label]
    green_bottom = [0 if i not in cell_trans or green_chan not in cell_trans[i] else cell_trans[i][green_chan] - 1 for i
                    in cell_label]

    uncolored = [frames_No for i in range(len(cell_label))]
    # TODO Add orange (red + green).
    # FIXME Why can green appear after red? Red and green should stay until the end of detection(?).

    N = len(uncolored)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    # plot tracked cells
    p0 = plt.bar(ind, uncolored, width, color='grey')

    # remove bars in frames where cells are not yet discovered
    # p1 = plt.bar(ind, first_appeared_filter, width, color='white')

    p2 = plt.bar(ind, green, width,
                 bottom=green_bottom, color='green')

    p3 = plt.bar(ind, red, width,
                 bottom=red_bottom, color='red')
    # red before green WA
    p4 = plt.bar(ind, green_if_red_before, width, bottom=green_bottom, color='green')

    # plot tracking "holes", i.e. paint bar in white, in frames where we lost track of cell
    for i in range(frames_No):
        missing_filter = [0 if (i + 1) in cell_frames[cell] else 1 for cell in cell_label]
        # if i == 5:
        plt.bar(ind, missing_filter, width, bottom=np.full(len(cell_label), i), color='white')

    plt.ylabel('Frame No.')
    plt.title('Cell Kinematics Visualisation')
    plt.xticks(ind, [str(cell_label[i]) for i in range(len(uncolored))])
    plt.yticks(np.arange(0, frames_No, 1), np.arange(1, frames_No + 1, 1))
    plt.legend((p0[0], p2[0], p3[0]), ('Uncolored', 'fitc', 'PI'))

    plt.show()
