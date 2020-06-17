import numpy as np
import random
import sklearn.mixture
import math

def get_random_tiles(width, height, level=1, tile_size=1024, count=5):
    tiles = []
    tiles_x = (width // tile_size) + 1
    tiles_y = (height // tile_size) + 1
    for i in range(level):
        tiles_x = math.ceil(tiles_x / 2)
        tiles_y = math.ceil(tiles_y / 2)

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
    center_x = tiles_x // 2
    center_y = tiles_y // 2
    tiles.append((center_x, center_y))

    for x in range(x_low, x_high+1):
        for y in range(y_low, y_high+1):
            if x != center_x or y != center_y:
                available.append((x, y))

    for i in range(count-1):
        if len(available) == 0:
            break
        i = random.randint(0, len(available)-1)
        random_tile = available.pop(i)
        tiles.append(random_tile)

    return tiles

def _smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def _local_minimas(arr, window=5):
    minimas = []
    for i in range(len(arr)-window):
        is_minima = True
        for l in range(i, i+window//2):
            if arr[l] <= arr[l+1]:
                is_minima = False

        for r in range(i+1+(window//2), i+window):
            if arr[r] <= arr[r-1]:
                is_minima = False

        if is_minima:
            minimas.append(i)

    return minimas

def calc_histogram(data, num_bins=255, range_max=65535):
    hist, bins = np.histogram(data.flatten(), bins=num_bins, range=(0, range_max))
    return hist, bins

def calc_min_max(histogram, bins, threshold, range_max=65535, smooth=10):
    h = _smooth(histogram, smooth)
    minimas = _local_minimas(h, window=5)
    max_val = range_max
    max_limit = threshold * sum(histogram)

    largest_bin = np.amax(h)
    largest_bin_idx = np.where(h == largest_bin)

    min_val_idx = largest_bin_idx[0][0] + 1

    if len(minimas) > 0:
        if minimas[0] > max_limit:
            min_val_idx = minimas[0]

    min_val = bins[min_val_idx]

    for i in range(min_val_idx+1, len(bins)-1):
        bin_size = h[i]
        if max_val == range_max and bin_size < max_limit:
            max_val = bins[i]
            break

    max_val = round(max_val)
    min_val = round(min_val)
    return min_val / range_max, max_val / range_max

def gaussian(data, n_components=3, n_sigmas=2, subsampling=1, range_max=65535):
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='spherical')
    d = data.reshape(-1, 1)
    subsampled = d[::subsampling]
    subsampled[subsampled == 0] = 1
    subsampled = np.log(subsampled)

    gmm.fit(subsampled)

    min_max = gmm.means_ + [-1, 1] * gmm.covariances_.reshape(-1, 1)**0.5*n_sigmas
    i = np.argmax(min_max)
    min_max = min_max.flatten()

    min_value = min_max[i-1]
    max_value = min_max[i]

    min_value = math.exp(min_value)
    max_value = math.exp(max_value)

    min_value = max(0, min_value)
    max_value = min(range_max, max_value)

    return min_value / range_max, max_value / range_max

