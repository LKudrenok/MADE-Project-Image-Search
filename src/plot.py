from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .search import Neighbour


def draw_windows_on_one_image(image_filepath: str, neighbours: List[Neighbour], folder: Path, show: bool = True):
    fig = plt.figure(figsize=(18, 8))
    image = plt.imread(image_filepath)
    plt.imshow(image)
    for n in neighbours:
        left, top, right, bottom = n.window
        plt.gca().add_patch(patches.Rectangle((left, top), right - left, bottom - top, linewidth=3, edgecolor='r', facecolor='none'))
        # plt.annotate(round(n.score, 1), (left, top), color='blue')
    plt.axis('off')
    filepath = Path(folder, Path(image_filepath).name)
    plt.savefig(filepath, transparent=False)
    if show:
        plt.show()
    return fig
