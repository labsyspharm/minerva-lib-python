'''Compare crop results with expected output'''

import pytest
import numpy as np
from minerva_lib.crop import scale_image_nearest_neighbor
from minerva_lib.crop import get_optimum_pyramid_level
from minerva_lib.crop import scale_by_pyramid_level
from minerva_lib.crop import select_tiles
from minerva_lib.crop import get_subregion
from minerva_lib.crop import get_position
from minerva_lib.crop import stitch_tile
from minerva_lib.crop import stitch_tiles
from minerva_lib import skimage_inline as ski


@pytest.fixture
def color_black():
    return np.array([0, 0, 0], dtype=np.float32)


@pytest.fixture
def color_red():
    return np.array([1, 0, 0], dtype=np.float32)


@pytest.fixture
def color_green():
    return np.array([0, 1, 0], dtype=np.float32)


@pytest.fixture
def color_half_mean_red_green():
    return np.array([0.25, 0.25, 0], dtype=np.float32)


@pytest.fixture
def color_blue():
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture
def color_magenta():
    return np.array([0, 1, 0], dtype=np.float32)


@pytest.fixture
def color_half_mean_blue_magenta():
    return np.array([0, 0.25, 0.25], dtype=np.float32)


@pytest.fixture
def color_cyan():
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture
def color_orange():
    return np.array([1, .5, 0], dtype=np.float32)


@pytest.fixture
def color_half_mean_cyan_orange():
    return np.array([0.25, 0.38, 0.25], dtype=np.float32)


@pytest.fixture
def origin_zero():
    return [0, 0]


@pytest.fixture
def indices_1_1():
    return [1, 1]


@pytest.fixture
def tile_size_2x2():
    return [2, 2]


@pytest.fixture
def tile_subregion_2x2():
    return [
        [0, 0],
        [2, 2]
    ]


@pytest.fixture
def level0_shape_6x6():
    return [6, 6]


@pytest.fixture
def scaled_shape_4x4():
    return [4, 4]


@pytest.fixture
def level1_shape_3x3():
    return [3, 3]


@pytest.fixture
def max_4():
    return 4


@pytest.fixture
def num_levels_2():
    return 2


@pytest.fixture
def round_up():
    return True


@pytest.fixture
def round_down():
    return False


@pytest.fixture
def level0_tiles_green_mask():
    ''' Nine 2x2 pixel tiles, green channel
    '''
    return [
        [
            np.array([
                [0, 255],
                [0, 0]
            ], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8)
        ],
    ] * 3


@pytest.fixture
def level0_tiles_red_mask():
    ''' Nine 2x2 pixel tiles, red channel
    '''
    return [
        [
            np.array([
                [0, 0],
                [255, 0]
            ], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8)
        ],
    ] * 3


@pytest.fixture
def level0_tiles_magenta_mask():
    ''' Nine 2x2 pixel tiles, magenta channel
    '''
    return [
        [
            np.zeros([2, 2], dtype=np.uint8),
            np.array([
                [0, 255],
                [0, 0]
            ], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8)
        ],
    ] * 3


@pytest.fixture
def level0_tiles_blue_mask():
    ''' Nine 2x2 pixel tiles, blue channel
    '''
    return [
        [
            np.zeros([2, 2], dtype=np.uint8),
            np.array([
                [0, 0],
                [255, 0]
            ], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8)
        ],
    ] * 3


@pytest.fixture
def level0_tiles_orange_mask():
    ''' Nine 2x2 pixel tiles, orange channel
    '''
    return [
        [
            np.zeros([2, 2], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8),
            np.array([
                [0, 255],
                [0, 0]
            ], dtype=np.uint8)
        ],
    ] * 3


@pytest.fixture
def level0_tiles_cyan_mask():
    ''' Nine 2x2 pixel tiles, cyan channel
    '''
    return [
        [
            np.zeros([2, 2], dtype=np.uint8),
            np.zeros([2, 2], dtype=np.uint8),
            np.array([
                [0, 0],
                [255, 0]
            ], dtype=np.uint8)
        ],
    ] * 3


@pytest.fixture
def level0_tiles(color_black, color_red, color_green, color_blue,
                 color_magenta, color_cyan, color_orange):
    ''' Nine 2x2 pixel tiles in a square checkerboard
    '''
    return [
        [
            np.array([
                [color_black, color_green],
                [color_red, color_black]
            ]),
            np.array([
                [color_black, color_magenta],
                [color_blue, color_black]
            ]),
            np.array([
                [color_black, color_orange],
                [color_cyan, color_black]
            ])
        ],
    ] * 3


@pytest.fixture
def level0_tile_list():

    return [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2]
    ]


@pytest.fixture
def level0_scaled_4x4(color_black, color_red, color_blue,
                      color_magenta, color_orange):
    ''' One 4x4 image scaled from the 6x6 pixels of level 0
    '''
    row_0 = [
        color_black, color_black,
        color_magenta, color_orange
    ]
    row_1 = [
        color_red, color_blue,
        color_black, color_black
    ]
    return np.array([
        row_0, row_0,
        row_1, row_1
    ])


@pytest.fixture
def level0_stitched_green_mask():
    ''' One 6x6 pixel green channel stitched from nine tiles
    '''
    row_0 = [
        0, 255, 0, 0, 0, 0
    ]
    row_1 = [
        0, 0, 0, 0, 0, 0
    ]
    return np.array([row_0, row_1] * 3, dtype=np.uint8)


@pytest.fixture
def level0_stitched(color_black, color_red, color_green, color_blue,
                    color_magenta, color_cyan, color_orange):
    ''' One 6x6 pixel image stitched from nine tiles
    '''
    row_0 = [
        color_black, color_green,
        color_black, color_magenta,
        color_black, color_orange
    ]
    row_1 = [
        color_red, color_black,
        color_blue, color_black,
        color_cyan, color_black,
    ]
    return np.array([row_0, row_1] * 3)


@pytest.fixture
def level1_tiles(color_half_mean_red_green,
                 color_half_mean_blue_magenta,
                 color_half_mean_cyan_orange):
    ''' One full + two half + one fourth 2x2 pixel tiles
        as the half-resolution representation of level0
        using linear interpolation.
    '''

    return [
        [
            np.array([
                [
                    color_half_mean_red_green,
                    color_half_mean_blue_magenta
                ]
            ] * 2),
            np.array([
                [
                    color_half_mean_cyan_orange
                ]
            ] * 2),
        ],
        [
            np.array([
                [
                    color_half_mean_red_green,
                    color_half_mean_blue_magenta
                ]
            ]),
            np.array([
                [
                    color_half_mean_cyan_orange
                ]
            ]),
        ]
    ]


@pytest.fixture
def level1_tile_list():

    return [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]


@pytest.fixture
def level1_stitched(color_half_mean_red_green,
                    color_half_mean_blue_magenta,
                    color_half_mean_cyan_orange):
    ''' One 3x3 pixel image of a half-scaled level0
        using linear interpolation
    '''
    row = [
        color_half_mean_red_green,
        color_half_mean_blue_magenta,
        color_half_mean_cyan_orange
    ]
    return np.array([row, row, row])


def test_scale_image_level0_4x4(level0_stitched, level0_scaled_4x4):
    ''' Test downsampling level0 to 2/3 size without interpolation '''

    expected = level0_scaled_4x4

    result = scale_image_nearest_neighbor(level0_stitched, 2 / 3)

    np.testing.assert_allclose(expected, result)


def test_get_optimum_pyramid_level_up(level0_shape_6x6, num_levels_2,
                                      max_4, round_up):
    ''' Test rounding up to pyramid level contained by maximum'''

    expected = 1

    result = get_optimum_pyramid_level(level0_shape_6x6, num_levels_2,
                                       max_4, round_up)

    assert expected == result


def test_get_optimum_pyramid_level_down(level0_shape_6x6, num_levels_2,
                                        max_4, round_down):
    ''' Test rounding down to pyramid level containing maximum'''

    expected = 0

    result = get_optimum_pyramid_level(level0_shape_6x6, num_levels_2,
                                       max_4, round_down)

    assert expected == result


def test_scale_by_pyramid_level0(level0_shape_6x6):
    ''' Test keeping level 0 coordinates unchanged'''

    expected = np.array(level0_shape_6x6, dtype=np.int64)

    result = scale_by_pyramid_level(level0_shape_6x6, 0)

    np.testing.assert_array_equal(expected, result)


def test_scale_by_pyramid_level1(level0_shape_6x6, level1_shape_3x3):
    ''' Test scaling level 0 coordinates to level 1 coordinates'''

    expected = np.array(level1_shape_3x3, dtype=np.int64)

    result = scale_by_pyramid_level(level0_shape_6x6, 1)

    np.testing.assert_array_equal(expected, result)


def test_select_tiles_level0(origin_zero, tile_size_2x2,
                             level0_shape_6x6, level0_tile_list):
    ''' Test selecting all level 0 tiles'''

    expected = level0_tile_list

    result = select_tiles(tile_size_2x2, origin_zero, level0_shape_6x6)

    np.testing.assert_array_equal(expected, result)


def test_select_tiles_level1(origin_zero, tile_size_2x2,
                             level1_shape_3x3, level1_tile_list):
    ''' Test selecting all level 1 tiles'''

    expected = level1_tile_list

    result = select_tiles(tile_size_2x2, origin_zero, level1_shape_3x3)

    np.testing.assert_array_equal(expected, result)


def test_get_subregion_1_1(origin_zero, tile_size_2x2,
                           level1_shape_3x3, indices_1_1):
    ''' Test subregion in 2x2 tile at 1,1 for 3x3 shape'''

    expected = [
        [0, 0],
        [1, 1]
    ]

    result = get_subregion(indices_1_1, tile_size_2x2,
                           origin_zero, level1_shape_3x3)

    np.testing.assert_array_equal(expected, result)


def test_get_position_1_1(origin_zero, tile_size_2x2, indices_1_1):
    ''' Test position for 2x2 tile at 1, 1'''

    expected = [2, 2]

    result = get_position(indices_1_1, tile_size_2x2, origin_zero)

    np.testing.assert_array_equal(expected, result)


def test_stitch_tile_level0(level0_stitched_green_mask, level0_shape_6x6,
                            level0_tiles_green_mask, tile_subregion_2x2):
    ''' Test stitching green channel in level 0 using three tiles'''

    expected = level0_stitched_green_mask

    subregion = tile_subregion_2x2
    tiles = level0_tiles_green_mask
    result = np.zeros(level0_shape_6x6)

    result = stitch_tile(result, subregion, [0, 0], tiles[0][0])
    result = stitch_tile(result, subregion, [0, 2], tiles[1][0])
    result = stitch_tile(result, subregion, [0, 4], tiles[2][0])

    np.testing.assert_array_equal(expected, result)


def test_stitch_tiles_level0(level0_tiles_green_mask,
                             level0_tiles_red_mask,
                             level0_tiles_magenta_mask,
                             level0_tiles_blue_mask,
                             level0_tiles_orange_mask,
                             level0_tiles_cyan_mask,
                             color_red, color_green, color_blue,
                             color_magenta, color_cyan, color_orange,
                             tile_size_2x2, origin_zero, level0_shape_6x6,
                             level0_stitched):
    ''' Test stitching all tiles for all channels to render level 0'''

    expected = ski.adjust_gamma(level0_stitched, 1 / 2.2)

    result = stitch_tiles([{
        'min': 0,
        'max': 1,
        'indices': [0, 0],
        'image': level0_tiles_green_mask[0][0],
        'color': color_green
    }, {
        'min': 0,
        'max': 1,
        'indices': [0, 1],
        'image': level0_tiles_green_mask[1][0],
        'color': color_green
    }, {
        'min': 0,
        'max': 1,
        'indices': [0, 2],
        'image': level0_tiles_green_mask[2][0],
        'color': color_green
    }, {
        'min': 0,
        'max': 1,
        'indices': [0, 0],
        'image': level0_tiles_red_mask[0][0],
        'color': color_red
    }, {
        'min': 0,
        'max': 1,
        'indices': [0, 1],
        'image': level0_tiles_red_mask[1][0],
        'color': color_red
    }, {
        'min': 0,
        'max': 1,
        'indices': [0, 2],
        'image': level0_tiles_red_mask[2][0],
        'color': color_red
    }, {
        'min': 0,
        'max': 1,
        'indices': [1, 0],
        'image': level0_tiles_magenta_mask[0][1],
        'color': color_magenta
    }, {
        'min': 0,
        'max': 1,
        'indices': [1, 1],
        'image': level0_tiles_magenta_mask[1][1],
        'color': color_magenta
    }, {
        'min': 0,
        'max': 1,
        'indices': [1, 2],
        'image': level0_tiles_magenta_mask[2][1],
        'color': color_magenta
    }, {
        'min': 0,
        'max': 1,
        'indices': [1, 0],
        'image': level0_tiles_blue_mask[0][1],
        'color': color_blue
    }, {
        'min': 0,
        'max': 1,
        'indices': [1, 1],
        'image': level0_tiles_blue_mask[1][1],
        'color': color_blue
    }, {
        'min': 0,
        'max': 1,
        'indices': [1, 2],
        'image': level0_tiles_blue_mask[2][1],
        'color': color_blue
    }, {
        'min': 0,
        'max': 1,
        'indices': [2, 0],
        'image': level0_tiles_orange_mask[0][2],
        'color': color_orange
    }, {
        'min': 0,
        'max': 1,
        'indices': [2, 1],
        'image': level0_tiles_orange_mask[1][2],
        'color': color_orange
    }, {
        'min': 0,
        'max': 1,
        'indices': [2, 2],
        'image': level0_tiles_orange_mask[2][2],
        'color': color_orange
    }, {
        'min': 0,
        'max': 1,
        'indices': [2, 0],
        'image': level0_tiles_cyan_mask[0][2],
        'color': color_cyan
    }, {
        'min': 0,
        'max': 1,
        'indices': [2, 1],
        'image': level0_tiles_cyan_mask[1][2],
        'color': color_cyan
    }, {
        'min': 0,
        'max': 1,
        'indices': [2, 2],
        'image': level0_tiles_cyan_mask[2][2],
        'color': color_cyan
    }], tile_size_2x2, origin_zero, level0_shape_6x6)

    np.testing.assert_allclose(expected, result)
