import numpy as np
import random

def get_random_tiles(width, height, level=1, tile_size=1024, count=4):
    tiles = []
    tiles_x = (width // tile_size) + 1
    tiles_y = (height // tile_size) + 1
    for i in range(level):
        tiles_x = tiles_x // 2
        tiles_y = tiles_y // 2

    # Generate random tile coordinates from the image center,
    # ignoring the edge tiles, which have higher probability of
    # being background.

    #  .........         . = Ignored tile
    #  .ooXoooo.         o = Elective tiles
    #  .oooXoXo.         X = Chosen tiles
    #  .oooXooo.
    #  .........

    x_low = 1 if tiles_x >= 4 else 0
    x_high = tiles_x - 2 if tiles_x >= 4 else tiles_x - 1
    y_low = 1 if tiles_y >= 4 else 0
    y_high = tiles_y - 2 if tiles_y >= 4 else tiles_y - 1

    available = []
    for x in range(x_low, x_high+1):
        for y in range(y_low, y_high+1):
            available.append((x, y))

    for i in range(count):
        if len(available) == 0:
            break
        i = random.randint(0, len(available)-1)
        random_tile = available.pop(i)
        tiles.append(random_tile)

    return tiles

def calc_min_max(data, threshold, num_bins=100, abs_max=65535):
    h, bins = np.histogram(data.flatten(), bins=num_bins)
    max_val = None
    min_val = bins[1]

    for i, sum in reversed(list(enumerate(h))):
        if max_val is None and sum > threshold:
            max_val = bins[i]

    if max_val is None:
        max_val = abs_max

    min_val = round(min_val)
    max_val = round(max_val)
    return min_val / abs_max, max_val / abs_max