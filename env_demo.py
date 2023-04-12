import os
import numpy as np
import imageio.v2 as imageio
import seaborn as sns
import matplotlib.pyplot as plt

"""
0: agricultrual
1: on-farm recharge
2: habitat
3: wetland
"""

def gen_map(idx):
    if 0 in map:
        idx1 = map_idx // map_size[1]
        idx2 = map_idx % map_size[1]
        assert map[idx1, idx2] == 0
        map[idx1, idx2] = np.random.randint(1, 4)


def plot_map(year):
    os.makedirs("demo", exist_ok=True)
    plt.figure(figsize=(6, 3), dpi=200)
    name_annot = [[name_dict[i] for i in row] for row in map]
    ax = sns.heatmap(map, vmin=0, vmax=3, cmap=color_map, annot=name_annot, fmt='', xticklabels=False, yticklabels=False,
                     cbar=False, annot_kws={"size": 10})
    plt.title(f"Year {year}")
    # plt.show()
    fname = f"./demo/map_{year}.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    frames.append(fname)


if __name__ == "__main__":
    name_dict = {0: 'Ag', 1: 'Hbt', 2: 'OFR', 3: 'Wl'}
    color_map = sns.color_palette("YlGnBu", 4)
    map_size = (2, 5)
    map_idx = 0
    map = np.zeros(map_size, dtype=int)
    frames = []

    for i in range(20):
        plot_map(i)
        if i in [2, 5, 7, 10, 11, 14, 16, 18]:
            gen_map(map_idx)
            map_idx += 1

    with imageio.get_writer('./demo/map.mp4', fps=1.5) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    for frame in frames:
        os.remove(frame)
