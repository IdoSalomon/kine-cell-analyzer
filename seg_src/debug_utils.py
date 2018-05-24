import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

    green_red_same_frame = [0 if i not in cell_trans or green_chan not in cell_trans[i] or red_chan not in cell_trans[i]
                                 or (cell_trans[i][green_chan] != cell_trans[i][red_chan]) else 1 for i in cell_label]

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

    p5 = plt.bar(ind, green_red_same_frame, width, bottom=green_bottom, color='yellow')

    # plot tracking "holes", i.e. paint bar in white, in frames where we lost track of cell
    for i in range(frames_No):
        missing_filter = [0 if (i + 1) in cell_frames[cell] else 1 for cell in cell_label]
        # if i == 5:
        plt.bar(ind, missing_filter, width, bottom=np.full(len(cell_label), i), color='white')

    plt.ylabel('Frame No.')
    plt.title('Cell Kinematics Visualisation')
    plt.xticks(ind, [str(cell_label[i]) for i in range(len(uncolored))])
    plt.yticks(np.arange(0, frames_No, 1), np.arange(1, frames_No + 1, 1))
    plt.legend((p0[0], p2[0], p3[0], p5[0]), ('Uncolored', 'fitc', 'PI', 'fitc and PI - same frame'))

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

def setup_ground_truth(frame):
    phase = np.copy(frame.images["phase"])
    rand_cells = [random.randint(1, 720) for i in range(100)]
    for cell in rand_cells:
        cv2.circle(phase, frame.cells[cell].centroid, radius=10, color=0)
        cv2.putText(phase, str(cell), frame.cells[cell].centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, 255, 1)
    phase = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("images\\L136\\A2\\4\\ground\\ground.tif", np.uint8(phase))
