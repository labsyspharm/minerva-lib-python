'''Compare crop results with expected output'''

import pytest
import numpy as np
from pathlib import Path
from inspect import currentframe, getframeinfo
from minerva_lib.crop import scale_image_nearest_neighbor
from minerva_lib.crop import get_optimum_pyramid_level
from minerva_lib.crop import scale_by_pyramid_level
from minerva_lib.crop import validate_region_bounds
from minerva_lib.crop import get_tile_start
from minerva_lib.crop import get_tile_count
from minerva_lib.crop import select_tiles
from minerva_lib.crop import get_subregion
from minerva_lib.crop import get_position
from minerva_lib.crop import stitch_tile
from minerva_lib.crop import stitch_tiles
from minerva_lib import skimage_inline as ski


@pytest.fixture
def dirname():
    filename = getframeinfo(currentframe()).filename
    return Path(filename).resolve().parent


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
def origin_negative():
    return [-1, 0]


@pytest.fixture
def origin_zero():
    return [0, 0]


@pytest.fixture
def origin_0_1():
    return [0, 1]


@pytest.fixture
def shape_0x0():
    return [0, 0]


@pytest.fixture
def indices_1_1():
    return [1, 1]


@pytest.fixture
def indices_3_3():
    return [3, 3]


@pytest.fixture
def tile_shape_2x2():
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
def real_tiles_red_mask(dirname):
    return [
        [
            np.load(Path(dirname, 'data/red/0/0/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/1/0/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/2/0/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/3/0/tile.npy').resolve())
        ],
        [
            np.load(Path(dirname, 'data/red/0/1/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/1/1/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/2/1/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/3/1/tile.npy').resolve())
        ],
        [
            np.load(Path(dirname, 'data/red/0/2/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/1/2/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/2/2/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/3/2/tile.npy').resolve())
        ],
        [
            np.load(Path(dirname, 'data/red/0/3/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/1/3/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/2/3/tile.npy').resolve()),
            np.load(Path(dirname, 'data/red/3/3/tile.npy').resolve())
        ],
    ]


@pytest.fixture
def tile_shape_256x256():
    return [256, 256]


@pytest.fixture
def real_shape_1024x1024():
    return [1024, 1024]


@pytest.fixture
def real_stitched_with_gamma(dirname):
    ''' The blend.composite_channels function made the image using this:

    [{
        image: # from {URL}/C0-T0-Z0-L0-Y0-X0.png
        color: [1, 0, 0]
        min: 0
        max: 1
    },{
        image: # from {URL}/C1-T0-Z0-L0-Y0-X0.png
        color: [0, 1, 0]
        min: 0.006
        max: 0.024
    }]

    where {URL} is https://s3.amazonaws.com/minerva-test-images/png_tiles/
    '''

    return np.load(Path(dirname, 'data/red_green_normalized.npy').resolve())


@pytest.fixture
def real_tiles_green_mask(dirname):
    return [
        [
            np.load(Path(dirname, 'data/green/0/0/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/1/0/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/2/0/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/3/0/tile.npy').resolve())
        ],
        [
            np.load(Path(dirname, 'data/green/0/1/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/1/1/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/2/1/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/3/1/tile.npy').resolve())
        ],
        [
            np.load(Path(dirname, 'data/green/0/2/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/1/2/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/2/2/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/3/2/tile.npy').resolve())
        ],
        [
            np.load(Path(dirname, 'data/green/0/3/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/1/3/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/2/3/tile.npy').resolve()),
            np.load(Path(dirname, 'data/green/3/3/tile.npy').resolve())
        ],
    ]


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


def test_scale_image_invalid_factor(level0_stitched):
    ''' Test downsampling level0 to 0% fails '''

    with pytest.raises(ValueError):
        scale_image_nearest_neighbor(level0_stitched, 0)


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


def test_tile_start_1_1(indices_1_1, tile_shape_2x2):
    ''' Ensure correct lower bound for origin after first tile'''

    expected = indices_1_1

    result = get_tile_start(tile_shape_2x2, tile_shape_2x2)

    np.testing.assert_array_equal(expected, result)


def test_tile_count_3_3(origin_zero, indices_3_3,
                        level0_shape_6x6, tile_shape_2x2):
    ''' Ensure correct upper bound within level0 shape'''

    expected = indices_3_3

    result = get_tile_count(tile_shape_2x2, origin_zero,
                            level0_shape_6x6)

    np.testing.assert_array_equal(expected, result)


def test_validate_region_whole(origin_zero, level0_shape_6x6):
    ''' Ensure full region is validated '''

    result = validate_region_bounds(origin_zero, level0_shape_6x6,
                                    level0_shape_6x6)

    assert result


def test_validate_region_within(origin_0_1, tile_shape_2x2,
                                level0_shape_6x6):
    ''' Ensure partial region is validated '''

    result = validate_region_bounds(origin_0_1, tile_shape_2x2,
                                    level0_shape_6x6)

    assert result


def test_validate_region_exceeds(origin_0_1, level0_shape_6x6):
    ''' Ensure excessively large region is invalidated '''

    result = validate_region_bounds(origin_0_1, level0_shape_6x6,
                                    level0_shape_6x6)

    assert not result


def test_validate_region_empty(origin_zero, shape_0x0,
                               level0_shape_6x6):
    ''' Ensure empty region is invalidated '''

    result = validate_region_bounds(origin_zero, shape_0x0,
                                    level0_shape_6x6)

    assert not result


def test_validate_region_negative(origin_negative, tile_shape_2x2,
                                  level0_shape_6x6):
    ''' Ensure negative region is invalidated '''

    result = validate_region_bounds(origin_negative, tile_shape_2x2,
                                    level0_shape_6x6)

    assert not result


def test_select_tiles_level0(origin_0_1, tile_shape_2x2):
    ''' Ensure selection of two tiles for partial region '''

    expected = [
        [0, 0],
        [0, 1]
    ]

    result = select_tiles(tile_shape_2x2, origin_0_1, tile_shape_2x2)

    np.testing.assert_array_equal(expected, result)


def test_select_tiles_level1(origin_zero, tile_shape_2x2,
                             level1_shape_3x3, level1_tile_list):
    ''' Ensure selection of all avaialable tiles for full region '''

    expected = level1_tile_list

    result = select_tiles(tile_shape_2x2, origin_zero, level1_shape_3x3)

    np.testing.assert_array_equal(expected, result)


def test_get_subregion_1_1(origin_zero, tile_shape_2x2,
                           level1_shape_3x3, indices_1_1):
    ''' Ensure partial tile is selected when full tile unnecessary '''

    expected = [
        [0, 0],
        [1, 1]
    ]

    result = get_subregion(indices_1_1, tile_shape_2x2,
                           origin_zero, level1_shape_3x3)

    np.testing.assert_array_equal(expected, result)


def test_get_position_1_1(origin_zero, tile_shape_2x2, indices_1_1):
    ''' Test ability to position tile in middle of image '''

    expected = [2, 2]

    result = get_position(indices_1_1, tile_shape_2x2, origin_zero)

    np.testing.assert_array_equal(expected, result)


def test_stitch_tile_level0(level0_stitched_green_mask, level0_shape_6x6,
                            level0_tiles_green_mask, tile_subregion_2x2):
    ''' Test correct cropping of single channel without rendering '''

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
                             tile_shape_2x2, origin_zero, level0_shape_6x6,
                             level0_stitched):
    ''' Ensure expected rendering of multi-tile multi-channel image '''

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
    }], tile_shape_2x2, origin_zero, level0_shape_6x6)

    np.testing.assert_allclose(expected, result)


def test_stitch_tiles_real(real_tiles_green_mask,
                           real_tiles_red_mask,
                           color_red, color_green,
                           tile_shape_256x256, origin_zero,
                           real_shape_1024x1024,
                           real_stitched_with_gamma):
    ''' Ensure 1024 x 1024 image matches image rendered without tiling '''

    expected = real_stitched_with_gamma

    inputs = []

    for i in range(0, 4):
        for j in range(0, 4):
            inputs += [{
                'min': 0.006,
                'max': 0.024,
                'indices': [i, j],
                'image': real_tiles_green_mask[j][i],
                'color': color_green
            }, {
                'min': 0,
                'max': 1,
                'indices': [i, j],
                'image': real_tiles_red_mask[j][i],
                'color': color_red
            }]

    result = stitch_tiles(inputs, tile_shape_256x256,
                          origin_zero, real_shape_1024x1024)

    np.testing.assert_allclose(expected, np.uint8(255*result))
