'''Compare crop results with expected output'''

import pytest
import numpy as np
from pathlib import Path
from inspect import currentframe, getframeinfo
from minerva_lib.crop import scale_image_nearest_neighbor, get_tile_start, \
                             get_optimum_pyramid_level, get_tile_count, \
                             scale_by_pyramid_level, select_tiles, \
                             validate_region_bounds, select_subregion, \
                             select_position, composite_subtile, \
                             composite_subtiles, extract_subtile
from minerva_lib import skimage_inline as ski


@pytest.fixture(scope='module')
def dirname():
    filename = getframeinfo(currentframe()).filename
    return Path(filename).resolve().parent


@pytest.fixture(scope='module')
def color_black():
    return np.array([0, 0, 0], dtype=np.float32)


@pytest.fixture(scope='module')
def color_red():
    return np.array([1, 0, 0], dtype=np.float32)


@pytest.fixture(scope='module')
def color_green():
    return np.array([0, 1, 0], dtype=np.float32)


@pytest.fixture(scope='module')
def color_half_mean_red_green():
    return np.array([0.25, 0.25, 0], dtype=np.float32)


@pytest.fixture(scope='module')
def color_blue():
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture(scope='module')
def color_magenta():
    return np.array([0, 1, 0], dtype=np.float32)


@pytest.fixture(scope='module')
def color_half_mean_blue_magenta():
    return np.array([0, 0.25, 0.25], dtype=np.float32)


@pytest.fixture(scope='module')
def color_cyan():
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture(scope='module')
def color_orange():
    return np.array([1, .5, 0], dtype=np.float32)


@pytest.fixture(scope='module')
def color_half_mean_cyan_orange():
    return np.array([0.25, 0.38, 0.25], dtype=np.float32)


@pytest.fixture(scope='module')
def origin_negative():
    return (0, -1)


@pytest.fixture(scope='module')
def origin_zero():
    return (0, 0)


@pytest.fixture(scope='module')
def origin_1_0():
    return (1, 0)


@pytest.fixture(scope='module')
def shape_0x0():
    return (0, 0)


@pytest.fixture(scope='module')
def indices_0_0():
    return (0, 0)


@pytest.fixture(scope='module')
def indices_1_1():
    return (1, 1)


@pytest.fixture(scope='module')
def indices_3_3():
    return (3, 3)


@pytest.fixture(scope='module')
def tile_shape_2x2():
    return (2, 2)


@pytest.fixture(scope='module')
def tile_subregion_2x2():
    return [
        (0, 0),
        (2, 2)
    ]


@pytest.fixture(scope='module')
def level0_shape_6x6():
    return (6, 6)


@pytest.fixture(scope='module')
def scaled_shape_4x4():
    return (4, 4)


@pytest.fixture(scope='module')
def level1_shape_3x3():
    return (3, 3)


@pytest.fixture(scope='module')
def max_4():
    return 4


@pytest.fixture(scope='module')
def num_levels_2():
    return 2


@pytest.fixture(scope='module')
def prefer_higher():
    return True


@pytest.fixture(scope='module')
def prefer_lower():
    return False


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def tile_shape_1024x1024():
    return (1024, 1024)


@pytest.fixture(scope='module')
def hd_shape_1080x1920():
    return (1080, 1920)


@pytest.fixture(scope='module')
def hd_stitched(hd_shape_1080x1920, color_green):
    hd_h, hd_w = hd_shape_1080x1920
    return np.ones((hd_h, hd_w, 3)) * color_green


@pytest.fixture(scope='module')
def hd_tiles_green_mask():
    return [
        [
            np.ones((1024, 1024)),
            np.ones((1024, 896))
        ],
        [
            np.ones((56, 1024)),
            np.ones((56, 896))
        ]
    ]


@pytest.fixture(scope='module')
def tile_shape_256x256():
    return (256, 256)


@pytest.fixture(scope='module')
def real_shape_1024x1024():
    return (1024, 1024)


@pytest.fixture(scope='module')
def real_stitched_with_gamma(dirname):
    ''' Loads a local red/green composite image from a npy file

    The red/green composite image originates from a call to the
    blend.composite_channels function with the following arguments:

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

    where {URL} is https://s3.amazonaws.com/minerva-test-images/png_tiles
    '''

    return np.load(Path(dirname, 'data/red_green_normalized.npy').resolve())


@pytest.fixture(scope='module')
def real_tiles_green_mask(dirname):
    ''' Loads 256x256 px image tiles from npy files

    The tiles originate from a 1024x1024 px file at {URL}/C1-T0-Z0-L0-Y0-X0.png

    where {URL} is https://s3.amazonaws.com/minerva-test-images/png_tiles
    '''

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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def level0_stitched_green_rgba(color_black, color_green):
    ''' One 6x6 pixel green channel stitched from nine tiles
    '''
    row_0 = [
        color_black, color_green, color_black,
        color_black, color_black, color_black
    ]
    row_1 = [
        color_black, color_black, color_black,
        color_black, color_black, color_black
    ]
    return np.array([row_0, row_1] * 3, dtype=np.uint8)


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def level1_tile_list():

    return [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


@pytest.fixture(scope='module')
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


def test_get_optimum_pyramid_level_higher(level0_shape_6x6, num_levels_2,
                                          max_4, prefer_higher):
    ''' Test higher resolution than needed for output shape'''

    expected = 0

    result = get_optimum_pyramid_level(level0_shape_6x6, num_levels_2,
                                       max_4, prefer_higher)

    assert expected == result


def test_get_optimum_pyramid_level_lower(level0_shape_6x6, num_levels_2,
                                         max_4, prefer_lower):
    ''' Test lower resolution than needed for output shape'''

    expected = 1

    result = get_optimum_pyramid_level(level0_shape_6x6, num_levels_2,
                                       max_4, prefer_lower)

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


def test_validate_region_within(origin_1_0, tile_shape_2x2,
                                level0_shape_6x6):
    ''' Ensure partial region is validated '''

    result = validate_region_bounds(origin_1_0, tile_shape_2x2,
                                    level0_shape_6x6)

    assert result


def test_validate_region_exceeds(origin_1_0, level0_shape_6x6):
    ''' Ensure excessively large region is invalidated '''

    result = validate_region_bounds(origin_1_0, level0_shape_6x6,
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


def test_select_tiles_level0(origin_1_0, tile_shape_2x2):
    ''' Ensure selection of two tiles for partial region '''

    expected = [
        (0, 0),
        (1, 0)
    ]

    result = select_tiles(tile_shape_2x2, origin_1_0, tile_shape_2x2)

    np.testing.assert_array_equal(expected, result)


def test_select_tiles_level1(origin_zero, tile_shape_2x2,
                             level1_shape_3x3, level1_tile_list):
    ''' Ensure selection of all available tiles for full region '''

    expected = level1_tile_list

    result = select_tiles(tile_shape_2x2, origin_zero, level1_shape_3x3)

    np.testing.assert_array_equal(expected, result)


def test_select_subregion_1_1(origin_zero, tile_shape_2x2,
                              level1_shape_3x3, indices_1_1):
    ''' Ensure partial tile is selected when full tile unavailable '''

    expected = [
        (0, 0),
        (1, 1)
    ]

    result = select_subregion(indices_1_1, tile_shape_2x2,
                              origin_zero, level1_shape_3x3)

    np.testing.assert_array_equal(expected, result)


def test_extract_subtile_1_1(origin_1_0, tile_shape_2x2,
                             indices_0_0, level1_tiles):
    ''' Ensure partial tile is extracted when full tile unnecessary '''

    tile = level1_tiles[0][0]
    expected = tile[1:, :]

    result = extract_subtile(indices_0_0, tile_shape_2x2,
                             origin_1_0, tile_shape_2x2, tile)

    np.testing.assert_array_equal(expected, result)


def test_select_position_1_1(origin_zero, tile_shape_2x2, indices_1_1):
    ''' Test ability to position tile in middle of image '''

    expected = (2, 2)

    result = select_position(indices_1_1, tile_shape_2x2, origin_zero)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_composite(level0_tiles_red_mask, color_red,
                                     level0_tiles_green_mask, color_green,
                                     tile_shape_2x2):
    ''' Ensure compositing with existing content of stitched region'''

    first_tile = level0_tiles_red_mask[0][0]
    second_tile = level0_tiles_green_mask[0][0]

    expected = np.array([
        [[0, 0, 0], color_green],
        [color_red, [0, 0, 0]],
    ])

    result = np.zeros(tile_shape_2x2 + (3,))

    result = composite_subtile(result, first_tile, (0, 0),
                               color_red, 0, 1)
    result = composite_subtile(result, second_tile, (0, 0),
                               color_green, 0, 1)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_level0(level0_stitched_green_rgba, level0_shape_6x6,
                                  level0_tiles_green_mask, color_green):
    ''' Test correct cropping of single channel without rendering '''

    expected = level0_stitched_green_rgba

    tiles = level0_tiles_green_mask
    result = np.zeros(level0_shape_6x6 + (3,))

    result = composite_subtile(result, tiles[0][0], (0, 0),
                               color_green, 0, 1)
    result = composite_subtile(result, tiles[1][0], (2, 0),
                               color_green, 0, 1)
    result = composite_subtile(result, tiles[2][0], (4, 0),
                               color_green, 0, 1)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtiles_level0(level0_tiles_green_mask,
                                   level0_tiles_red_mask,
                                   level0_tiles_magenta_mask,
                                   level0_tiles_blue_mask,
                                   level0_tiles_orange_mask,
                                   level0_tiles_cyan_mask,
                                   color_red, color_green, color_blue,
                                   color_magenta, color_cyan, color_orange,
                                   tile_shape_2x2, origin_zero,
                                   level0_shape_6x6, level0_stitched):
    ''' Ensure expected rendering of multi-tile multi-channel image '''

    expected = ski.adjust_gamma(level0_stitched, 1 / 2.2)

    result = composite_subtiles([{
        'min': 0,
        'max': 1,
        'indices': (0, 0),
        'image': level0_tiles_green_mask[0][0],
        'color': color_green
    }, {
        'min': 0,
        'max': 1,
        'indices': (1, 0),
        'image': level0_tiles_green_mask[1][0],
        'color': color_green
    }, {
        'min': 0,
        'max': 1,
        'indices': (2, 0),
        'image': level0_tiles_green_mask[2][0],
        'color': color_green
    }, {
        'min': 0,
        'max': 1,
        'indices': (0, 0),
        'image': level0_tiles_red_mask[0][0],
        'color': color_red
    }, {
        'min': 0,
        'max': 1,
        'indices': (1, 0),
        'image': level0_tiles_red_mask[1][0],
        'color': color_red
    }, {
        'min': 0,
        'max': 1,
        'indices': (2, 0),
        'image': level0_tiles_red_mask[2][0],
        'color': color_red
    }, {
        'min': 0,
        'max': 1,
        'indices': (0, 1),
        'image': level0_tiles_magenta_mask[0][1],
        'color': color_magenta
    }, {
        'min': 0,
        'max': 1,
        'indices': (1, 1),
        'image': level0_tiles_magenta_mask[1][1],
        'color': color_magenta
    }, {
        'min': 0,
        'max': 1,
        'indices': (2, 1),
        'image': level0_tiles_magenta_mask[2][1],
        'color': color_magenta
    }, {
        'min': 0,
        'max': 1,
        'indices': (0, 1),
        'image': level0_tiles_blue_mask[0][1],
        'color': color_blue
    }, {
        'min': 0,
        'max': 1,
        'indices': (1, 1),
        'image': level0_tiles_blue_mask[1][1],
        'color': color_blue
    }, {
        'min': 0,
        'max': 1,
        'indices': (2, 1),
        'image': level0_tiles_blue_mask[2][1],
        'color': color_blue
    }, {
        'min': 0,
        'max': 1,
        'indices': (0, 2),
        'image': level0_tiles_orange_mask[0][2],
        'color': color_orange
    }, {
        'min': 0,
        'max': 1,
        'indices': (1, 2),
        'image': level0_tiles_orange_mask[1][2],
        'color': color_orange
    }, {
        'min': 0,
        'max': 1,
        'indices': (2, 2),
        'image': level0_tiles_orange_mask[2][2],
        'color': color_orange
    }, {
        'min': 0,
        'max': 1,
        'indices': (0, 2),
        'image': level0_tiles_cyan_mask[0][2],
        'color': color_cyan
    }, {
        'min': 0,
        'max': 1,
        'indices': (1, 2),
        'image': level0_tiles_cyan_mask[1][2],
        'color': color_cyan
    }, {
        'min': 0,
        'max': 1,
        'indices': (2, 2),
        'image': level0_tiles_cyan_mask[2][2],
        'color': color_cyan
    }], tile_shape_2x2, origin_zero, level0_shape_6x6)

    np.testing.assert_allclose(expected, result)


def test_composite_subtiles_nonsquare(hd_tiles_green_mask, origin_zero,
                                      color_green, tile_shape_1024x1024,
                                      hd_shape_1080x1920, hd_stitched):
    ''' Ensure non-square image is stitched correctly with square tiles'''

    expected = ski.adjust_gamma(hd_stitched, 1 / 2.2)

    inputs = []

    for y in range(0, 2):
        for x in range(0, 2):
            inputs += [{
                'min': 0,
                'max': 1,
                'indices': (y, x),
                'image': hd_tiles_green_mask[y][x],
                'color': color_green
            }]

    result = composite_subtiles(inputs, tile_shape_1024x1024,
                                origin_zero, hd_shape_1080x1920)

    np.testing.assert_allclose(expected, result)


def test_composite_subtiles_real(real_tiles_green_mask,
                                 real_tiles_red_mask,
                                 color_red, color_green,
                                 tile_shape_256x256, origin_zero,
                                 real_shape_1024x1024,
                                 real_stitched_with_gamma):
    ''' Ensure 1024 x 1024 image matches image rendered without tiling '''

    expected = real_stitched_with_gamma

    inputs = []

    for y in range(0, 4):
        for x in range(0, 4):
            inputs += [{
                'min': 0.006,
                'max': 0.024,
                'indices': (y, x),
                'image': real_tiles_green_mask[y][x],
                'color': color_green
            }, {
                'min': 0,
                'max': 1,
                'indices': (y, x),
                'image': real_tiles_red_mask[y][x],
                'color': color_red
            }]

    result = composite_subtiles(inputs, tile_shape_256x256,
                                origin_zero, real_shape_1024x1024)

    np.testing.assert_allclose(expected, np.uint8(255*result))
