import math
import os

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DBG_DIR = 'dbg'

col_map = {"GFP": "Greens", "FITC": "Greens", "fitc": "Greens", "pi": "Reds", "PI": "Reds", "PHASE": "gray",
           "phase": "gray", "TxRed": "Reds", "TRANS": "gray",
           "trans": "gray"}

Box = collections.namedtuple("Box", ["top", "bottom", "left", "right"])

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


def plot_kinematics(cell_frames, cell_trans, frames_No, red_chan='PI', green_chan='fitc', all_frames=True, rand=True,
                    sample_size=30):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(40, 8))  # 400 for full size

    width = 0.6  # the width of the bars: can also be len(x) sequence

    if all_frames:
        if not rand:
            context = cell_frames
        else:
            context = np.random.choice(list(cell_frames), sample_size)
    else:
        context = cell_trans

    # labels

    # first key - green change, secondary key - red change, third key - cell number
    sort_func = lambda cell: (
        frames_No if cell not in cell_trans or green_chan not in cell_trans[cell] else cell_trans[cell][green_chan],
        frames_No if cell not in cell_trans or red_chan not in cell_trans[cell] else cell_trans[cell][red_chan],
        cell)

    cell_label = [i for i in sorted(context, key=sort_func)][:20]

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
    green_and_red = [0 if i not in cell_trans or green_chan not in cell_trans[i] or
                          red_chan not in cell_trans[i]
                     else 1 + frames_No - cell_trans[i][red_chan] if cell_trans[i][green_chan] <= cell_trans[i][
        red_chan] else
    1 + frames_No - cell_trans[i][green_chan]
                     for i in cell_label]
    green_and_red_bottom = [red_bottom[i] if green_if_red_before[i] == 0 else green_bottom[i] for i in
                            range(len(green_bottom))]

    uncolored = [frames_No for i in range(len(cell_label))]

    N = len(uncolored)

    ind = np.arange(N)  # the x locations for the groups
    print('ha')
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

    p5 = plt.bar(ind,
                 green_and_red,
                 width,
                 bottom=green_and_red_bottom,
                 color='orange')

    # plot tracking "holes", i.e. paint bar in white, in frames where we lost track of cell
    for i in range(frames_No):
        missing_filter = [0 if (i + 1) in cell_frames[cell] else 1 for cell in cell_label]
        plt.bar(ind, missing_filter, width, bottom=np.full(len(cell_label), i), color='white')

    plt.ylabel('Frame no. (time)', fontweight='bold', size='x-large')
    plt.xlabel('Cell ID', fontweight='bold', size='x-large')
    plt.title('Single-Cell Dynamic Change of Parameters over Time', fontweight='bold', size=22)
    # plt.xticks(ind, [str(cell_label[i]) for i in range(len(uncolored))])
    plt.xticks(ind, ind + 1, fontweight='bold', size='x-large')
    plt.yticks(np.arange(0, frames_No, 1), np.arange(1, frames_No + 1, 1), fontweight='bold', size='x-large')
    plt.legend((p0[0], p2[0], p3[0], p5[0]), ('Unchanged', 'PI', 'fitc', 'PI+fitc'), loc=4,
               prop={'size': 'x-large', 'weight': 'bold'})

    plt.show()

def plot_quantative(cell_trans, frames_No, red_chan='PI', green_chan='fitc'):
    frame_no = 14
    red_chan = 'PI'
    green_chan = 'fitc'

    red_per_frame = [0] * frame_no
    green_per_frame = [0] * frame_no
    both = [0] * frame_no

    for cell in cell_trans:
        if green_chan in cell_trans[cell]:
            green_per_frame[cell_trans[cell][green_chan] - 1] += 1
        if red_chan in cell_trans[cell]:
            red_per_frame[cell_trans[cell][red_chan] - 1] += 1
        if red_chan in cell_trans[cell] and green_chan in cell_trans[cell]:
            both[cell_trans[cell][red_chan] - 1] += 1

    green_per_frame = np.cumsum(green_per_frame)
    red_per_frame = np.cumsum(red_per_frame)
    both = np.cumsum(both)

    ind = np.arange(frame_no)

    plt.plot(ind, red_per_frame, color='red')
    plt.plot(ind, green_per_frame, color='green')
    plt.plot(ind, both, color='yellow')


def create_flow_cyt_data(seq_frames, channels, cells_trans):
    rows = []
    for frame_id in seq_frames:
        frame = seq_frames[frame_id]
        cells = frame.cells
        """static_sample = random.sample(list(set(seq_frames[frame_id].cells.keys()) - set(cells_trans.keys())), len([val for val in seq_frames[frame_id].cells.keys() if val in cells_trans]))
        for cell_id in cells:
            if cell_id == 0:  # Skip background
                continue
            if cell_id in cells_trans or cell_id in static_sample:
                # Calculate intensities for all requested channels
                cell_intensities = []
                for channel in channels:
                    intensity = np.mean(cells[cell_id].pixel_values[channel])
                    cell_intensities.append(intensity)

                rows.append({'Frame': frame_id, 'Cell': cell_id, 'X': cell_intensities[0], 'Y': cell_intensities[1]})"""
        for cell_id in cells:
            if cell_id == 0:  # Skip background
                continue
            # Calculate intensities for all requested channels
            cell_intensities = []
            for channel in channels:
                intensity = np.mean(cells[cell_id].pixel_values[channel])
                cell_intensities.append(intensity)

            rows.append({'Frame': frame_id, 'Cell': cell_id, 'X': cell_intensities[0], 'Y': cell_intensities[1]})
    df = pd.DataFrame(rows)
    """sns.jointplot(x="Y", y="X", data=df[df['Frame'] == 9], kind="kde")
    sns.jointplot(x="Y", y="X", data=df[df['Frame'] == 9], kind="reg")


    sns.jointplot(x="Y", y="X", data=df[df['Frame'] == 13], kind="kde")
    sns.jointplot(x="Y", y="X", data=df[df['Frame'] == 13], kind="reg")

    sns.jointplot(x="Y", y="X", data=df[df['Frame'] == 14], kind="kde")"""

    df.to_csv('kmeans.csv')

    df_14 = df[df['Frame'] == 14]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df_14.reindex(columns=['X', 'Y']))
    #kmeans = (GaussianMixture(n_components=4, covariance_type="full", tol=0.001).fit(df_14.reindex(columns=['X', 'Y']))).predict(df_14.reindex(columns=['X', 'Y']))
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    x = df_14['Y']
    y = df_14['X']
    plt.scatter(x, y, c=kmeans.labels_)

    #plt.scatter(df.Y[df['Frame'] == 14], df.X[df['Frame'] == 14], s=0.2, alpha=1)

    plt.show()

    return df


# def setup_ground_truth(frame):
#     phase = np.copy(frame.images["phase"])
#     rand_cells = [random.randint(1, 720) for i in range(100)]
#     for cell in rand_cells:
#         cv2.circle(phase, frame.cells[cell].centroid, radius=10, color=0)
#         cv2.putText(phase, str(cell), frame.cells[cell].centroid,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.53, 255, 1)
#     phase = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)
#     cv2.imwrite("images\\L136\\A2\\4\\ground\\ground.tif", np.uint8(phase))


def calc_bounding_coords(tracked_img, cell):
    pxls = np.argwhere(tracked_img == cell)

    box = Box(min(pxls, key=lambda x: x[0])[0],
              max(pxls, key=lambda x: x[0])[0],
              min(pxls, key=lambda x: x[1])[1],
              max(pxls, key=lambda x: x[1])[1])
    return box


def expand_box(box, xmax, ymax, pad=2):
    expanded = Box(max(0, box.top - pad),
                   min(ymax, box.bottom + pad),
                   max(0, box.left - pad),
                   min(xmax, box.right + pad))
    return expanded


# Box = namedtuple("Box", ["top", "bottom", "left", "right"])
# bounding_box = Box(min(label_ind, key=lambda x: x[0]),
#                    max(label_ind, key=lambda x: x[0]),
#                    min(label_ind, key=lambda x: x[1]),
#                    max(label_ind, key=lambda x: x[1]))

def evaluate_accuracy(seq_frames, cell_frames, cell_trans, sample_size=20, channels=('fitc', 'PI')):
    # choose a random sample of cells
    cells = np.random.choice(list(cell_frames), sample_size)
    trans_labels = list(set(cells).intersection(set(cell_trans)))
    results = {label: {} for label in cells}

    # for each cell both in labels and cell_trans, validate transition frame
    for cell in cells:
        for chan in channels:
            answer = False
            while not answer:
                if cell in cell_trans and chan in cell_trans[cell]:
                    frame = cell_trans[cell][chan]
                    tracked_img = seq_frames[frame].tracked_img
                    box = calc_bounding_coords(tracked_img, cell)
                    box = expand_box(box, tracked_img.shape[1] - 1, tracked_img.shape[0] - 1)
                    cell_chan_img = seq_frames[frame].images[chan][box.top: box.bottom + 1, box.left: box.right + 1]

                    cell_mask_img = seq_frames[frame].con_comps[box.top: box.bottom + 1, box.left: box.right + 1]

                    # plot
                    fig = plt.figure()
                    fig.add_subplot(1, 2, 1)
                    plt.title('cell {} - frame {} - mask'.format(cell, frame))
                    plt.imshow(cell_mask_img, 'gray')

                    fig.add_subplot(1, 2, 2)
                    plt.title('cell {} - frame {} - channel: {}'.format(cell, frame, chan))
                    plt.imshow(cell_chan_img, col_map[chan], vmin=0, vmax=255)
                    plt.show()

                    line = input("[marked] has cell {} changed in {} channel in frame {}?\n".format(cell, chan,
                                                                                                    cell_trans[cell][
                                                                                                        chan]))
                else:
                    # print mask + channel for each frame
                    cell_frame_no = len(cell_frames[cell])
                    fig = plt.figure(figsize=(8, cell_frame_no * 2))
                    plt.tight_layout()
                    i = 1
                    for frame in sorted(cell_frames[cell]):
                        tracked_img = seq_frames[frame].tracked_img
                        box = calc_bounding_coords(tracked_img, cell)
                        box = expand_box(box, tracked_img.shape[1] - 1, tracked_img.shape[0] - 1)
                        cell_chan_img = seq_frames[frame].images[chan][box.top: box.bottom + 1, box.left: box.right + 1]
                        cell_mask_img = seq_frames[frame].con_comps[box.top: box.bottom + 1, box.left: box.right + 1]

                        fig.add_subplot(cell_frame_no, 2, i)
                        plt.title('cell {} - frame {} - mask'.format(cell, frame))
                        plt.imshow(cell_mask_img, 'gray')
                        i += 1

                        fig.add_subplot(cell_frame_no, 2, i)
                        plt.title('cell {} - frame {} - channel: {}'.format(cell, frame, chan))
                        plt.imshow(cell_chan_img, col_map[chan], vmin=0, vmax=255)  # no normalization
                        i += 1

                    plt.show()
                    line = input(
                        "[unmarked] has cell {} changed in channel {} during the sequence?\n".format(cell, chan))
                if line == 'y':
                    results[cell][chan] = True
                elif line == 'n':
                    results[cell][chan] = False
                else:
                    print("please enter 'y' for yes or 'n' for no")
                    continue
                answer = True

    # calcuate accuracy
    cells_no = len(cells)
    chan_no = len(channels)

    false_pos = sum([1 if results[cell][chan] and
                          (cell not in cell_trans or
                           chan not in cell_trans[cell])
                     else 0 for cell in cells for chan in channels])
    false_neg = sum([1 if not results[cell][chan] and
                          cell in cell_trans and
                          chan in cell_trans[cell]
                     else 0 for cell in cells for chan in channels])

    print("false positives: {}, false negatives: {}, accuracy = {}".format(false_pos, false_neg, (
                cells_no * chan_no - false_neg - false_pos) / float(cells_no * chan_no)))
