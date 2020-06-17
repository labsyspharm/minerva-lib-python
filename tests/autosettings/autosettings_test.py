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

def test_get_random_tiles_highest_level():
    tiles = autosettings.get_random_tiles(27545, 23109, level=5, tile_size=1024, count=1)
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile[0] == 0
    assert tile[1] == 0

def test_get_random_tiles_2nd_highest_level():
    tiles = autosettings.get_random_tiles(27545, 23109, level=4, tile_size=1024, count=4)
    assert len(tiles) == 4

def test_get_random_tiles_exact():
    tiles = autosettings.get_random_tiles(2048, 2048, level=0, tile_size=1024, count=4)
    assert len(tiles) == 4

    tiles = autosettings.get_random_tiles(2048, 2048, level=1, tile_size=1024, count=1)
    assert len(tiles) == 1

def test_calc_histogram_min_max():
    data = [(1/log(i)*10000)-1000 for i in range(2, 10000)]
    arr = np.array(data)
    h, b = autosettings.calc_histogram(arr, num_bins=100)
    min_val, max_val = autosettings.calc_min_max(h, b, threshold=0.002)
    assert min_val <= max_val
    assert 0 <= min_val <= 1
    assert 0 <= max_val <= 1

def test_gaussian():
    data = np.array([1000, 1100, 1200, 1400, 1400, 1200, 1100, 1000,
                     10000, 11000, 12000, 14000, 14000, 12000, 11000, 10000,
                     30000, 34000, 36000, 37000, 37000, 36000, 34000, 30000])
    min_val, max_val = autosettings.gaussian(data, n_components=3, n_sigmas=2)
    assert 0 <= min_val <= 1
    assert 0 <= max_val <= 1
