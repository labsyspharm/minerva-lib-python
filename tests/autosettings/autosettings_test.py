from minerva_lib import autosettings
from math import log
import numpy as np

def test_get_random_tiles():
    tiles = autosettings.get_random_tiles(15000, 10000, level=1, tile_size=1024, count=4)
    assert len(tiles) == 4
    for tile in tiles:
        assert tile[0] >= 1
        assert tile[0] <= 7
        assert tile[1] >= 1
        assert tile[1] <= 4

def test_get_random_tiles_single():
    tiles = autosettings.get_random_tiles(600, 400, level=0, tile_size=1024, count=1)
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile[0] == 0
    assert tile[1] == 0

def test_calc_histogram():
    data = [(1/log(i)*10000)-1000 for i in range(2, 10000)]
    arr = np.array(data)
    min_val, max_val = autosettings.calc_min_max(arr, 10, num_bins=100)
    assert min_val <= max_val
    assert 0 <= min_val <= 1
    assert 0 <= max_val <= 1
