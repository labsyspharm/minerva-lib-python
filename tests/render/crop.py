'''Compare crop results with expected output'''

import pytest
import numpy as np
from pathlib import Path
from inspect import currentframe, getframeinfo
from minerva_lib.render import (scale_image_nearest_neighbor,
                                get_region_first_grid,
                                get_optimum_pyramid_level,
                                get_region_grid_shape,
                                transform_coordinates_to_level, select_grids,
                                validate_region_bounds, select_subregion,
                                select_position, composite_subtile,
                                composite_subtiles, extract_subtile)
from tests.common.colors import (color_white, color_red, color_green,
                                 color_orange)

FILENAME = getframeinfo(currentframe()).filename
DIRNAME = Path(FILENAME).resolve().parent.parent


@pytest.fixture(scope='module')
def real_tiles_red_mask():
    return [
        [
            np.load(Path(DIRNAME, 'data/red/0/0/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/1/0/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/2/0/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/3/0/tile.npy').resolve())
        ],
        [
            np.load(Path(DIRNAME, 'data/red/0/1/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/1/1/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/2/1/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/3/1/tile.npy').resolve())
        ],
        [
            np.load(Path(DIRNAME, 'data/red/0/2/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/1/2/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/2/2/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/3/2/tile.npy').resolve())
        ],
        [
            np.load(Path(DIRNAME, 'data/red/0/3/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/1/3/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/2/3/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/red/3/3/tile.npy').resolve())
        ],
    ]


@pytest.fixture(scope='module')
def real_stitched_with_gamma():
    '''Loads a local red/green composite image from a npy file.

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

    return np.load(Path(DIRNAME, 'data/red_green_normalized.npy').resolve())


@pytest.fixture(scope='module')
def real_tiles_green_mask():
    '''Loads 256x256 px image tiles from npy files.

    The tiles originate from a 1024x1024 px file at {URL}/C1-T0-Z0-L0-Y0-X0.png
    where {URL} is https://s3.amazonaws.com/minerva-test-images/png_tiles.
    '''

    return [
        [
            np.load(Path(DIRNAME, 'data/green/0/0/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/1/0/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/2/0/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/3/0/tile.npy').resolve())
        ],
        [
            np.load(Path(DIRNAME, 'data/green/0/1/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/1/1/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/2/1/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/3/1/tile.npy').resolve())
        ],
        [
            np.load(Path(DIRNAME, 'data/green/0/2/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/1/2/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/2/2/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/3/2/tile.npy').resolve())
        ],
        [
            np.load(Path(DIRNAME, 'data/green/0/3/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/1/3/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/2/3/tile.npy').resolve()),
            np.load(Path(DIRNAME, 'data/green/3/3/tile.npy').resolve())
        ],
    ]


@pytest.fixture(scope='module')
def checker_4x4():
    '''One 6x6 pixel image stitched from nine tiles.'''

    return np.array([
        [[0, 0, 0], [1, 1, 1]] * 2,
        [[1, 1, 1], [0, 0, 0]] * 2,
    ] * 2)


def test_scale_image_aliasing(checker_4x4):
    '''Test downsampling to 3/4 size without interpolation.'''

    expected = np.array([
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
    ])

    result = scale_image_nearest_neighbor(checker_4x4, 3 / 4)

    np.testing.assert_allclose(expected, result)


def test_scale_image_asymetry(checker_4x4):
    '''Test downsampling only in y to 3/4 size without interpolation.'''

    expected = np.array([
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
    ])

    result = scale_image_nearest_neighbor(checker_4x4, (1, 3 / 4))

    np.testing.assert_allclose(expected, result)


def test_scale_image_invalid_factor():
    '''Test downsampling level0 to 0% fails.'''

    with pytest.raises(ValueError):
        scale_image_nearest_neighbor(np.array([
                                        [[0, 0, 0], [0, 0, 0]],
                                        [[0, 0, 0], [0, 0, 0]]
                                    ]), (0, 0))


def test_get_optimum_pyramid_level_higher():
    '''Test higher resolution than needed for output shape.'''

    expected = 0

    result = get_optimum_pyramid_level((6, 6), 2,
                                       4, True)

    assert expected == result


def test_get_optimum_pyramid_level_lower():
    '''Test lower resolution than needed for output shape'''

    expected = 1

    result = get_optimum_pyramid_level((6, 6), 2,
                                       4, False)

    assert expected == result


def test_transform_coordinates_full_scale():
    '''Test keeping level 0 coordinates unchanged'''

    expected = np.array((6, 6), dtype=np.int64)

    result = transform_coordinates_to_level((6, 6), 0)

    np.testing.assert_array_equal(expected, result)


def test_transform_coordinates_half_scale():
    '''Test scaling level 0 coordinates to level 1 coordinates'''

    expected = np.array((3, 3), dtype=np.int64)

    result = transform_coordinates_to_level((6, 6), 1)

    np.testing.assert_array_equal(expected, result)


def test_select_position_inner_tile():
    '''Test ability to position tile in middle of image.'''

    expected = (2, 2)

    result = select_position((1, 1), (2, 2), (0, 0))

    np.testing.assert_array_equal(expected, result)


def test_get_first_grid_inner_tile():
    '''Ensure correct lower bound for origin after first tile'''

    expected = (1, 1)

    result = get_region_first_grid((2, 2), (2, 2))

    np.testing.assert_array_equal(expected, result)


def test_get_grid_shape_clipped_tiles():
    '''Ensure correct upper bound within level0 shape.'''

    expected = (4, 4)

    result = get_region_grid_shape((2, 2), (3, 3),
                                   (6, 6))

    np.testing.assert_array_equal(expected, result)


def test_validate_region_whole():
    '''Ensure full region is validated.'''

    assert validate_region_bounds(
        (0, 0),
        (6, 6),
        (6, 6)
    )


def test_validate_region_within():
    '''Ensure partial region is validated.'''

    assert validate_region_bounds(
        (1, 0),
        (2, 2),
        (6, 6)
    )


def test_validate_region_exceeds():
    '''Ensure excessively large region is invalidated.'''

    assert not validate_region_bounds(
        (1, 0),
        (6, 6),
        (6, 6)
    )


def test_validate_region_empty():
    '''Ensure empty region is invalidated.'''

    assert not validate_region_bounds(
        (0, 0),
        (0, 0),
        (6, 6)
    )


def test_validate_region_negative():
    '''Ensure negative region is invalidated.'''

    assert not validate_region_bounds(
        (0, -1),
        (2, 2),
        (6, 6)
    )


def test_select_grids_sub_region():
    '''Ensure selection of two tiles for partial region.'''

    expected = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2)
    ]

    result = select_grids(
        (2, 2),
        (3, 3),
        (2, 2)
    )

    np.testing.assert_array_equal(expected, result)


def test_select_grids_full_region():
    '''Ensure selection of all available tiles for full region.'''

    expected = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    result = select_grids((2, 2), (0, 0), (3, 3))

    np.testing.assert_array_equal(expected, result)


def test_select_subregion_ceiling_tile():
    '''Ensure partial tile is selected when full tile unavailable.'''

    expected = [
        (0, 0),
        (1, 1)
    ]

    result = select_subregion((1, 1), (2, 2),
                              (0, 0), (3, 3))

    np.testing.assert_array_equal(expected, result)


def test_extract_subtile_clipped_tile():
    '''Ensure partial tile is extracted when full tile unnecessary.'''

    expected = np.array([
        [[1, 1, 1], [0, 0, 0]]
    ])

    result = extract_subtile((0, 0), (2, 2),
                             (1, 0), (2, 2),
                             np.array([
                                [[0, 0, 0], [1, 1, 1]],
                                [[1, 1, 1], [0, 0, 0]]
                             ]))

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_blending():
    '''Ensure compositing with existing content of stitched region'''

    expected = np.array([
        [[0, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 0]],
    ])

    result = composite_subtile(np.zeros((2, 2) + (3,)),
                               np.array([[0, 0], [255, 255]], dtype=np.uint8),
                               (0, 0), color_red, 0, 1)
    result = composite_subtile(result,
                               np.array([[0, 255], [0, 255]], dtype=np.uint8),
                               (0, 0), color_green, 0, 1)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_aligning():
    '''Test correct alignment when compositing of two tiles.'''

    expected = np.array([
        [[0] * 3, [1] * 3, [0] * 3],
        [[1] * 3, [0] * 3, [0] * 3],
        [[0] * 3, [1] * 3, [0] * 3]
    ])

    result = composite_subtile(np.zeros((3, 3) + (3,)),
                               np.array([[0, 255], [255, 0]], dtype=np.uint8),
                               (0, 0), color_white, 0, 1)
    result = composite_subtile(result, np.array([[0, 255]], dtype=np.uint8),
                               (2, 0), color_white, 0, 1)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_nonsquare():
    '''Stitch a non-square image from 1 full and 3 edge tiles.'''

    # Gamma correction not needed for fully saturated white
    expected = np.ones((1080, 1920, 3))

    result = composite_subtiles([{
        'min': 0,
        'max': 1,
        'grid': (0, 0),
        'image': 255 * np.ones((1024, 1024), dtype=np.uint8),
        'color': color_white
    }, {
        'min': 0,
        'max': 1,
        'grid': (1, 0),
        'image': 255 * np.ones((56, 1024), dtype=np.uint8),
        'color': color_white
    }, {
        'min': 0,
        'max': 1,
        'grid': (0, 1),
        'image': 255 * np.ones((1024, 896), dtype=np.uint8),
        'color': color_white
    }, {
        'min': 0,
        'max': 1,
        'grid': (1, 1),
        'image': 255 * np.ones((56, 896), dtype=np.uint8),
        'color': color_white
    }], (1024, 1024), (0, 0), (1080, 1920))

    np.testing.assert_allclose(expected, result)


def test_composite_subtile_gamma():
    '''Render a single tile testing gamma correction'''

    # Gamma correction not needed for fully saturated white
    expected = np.ones((1024, 1024, 3)) * [1,  0.72974005, 0]

    result = composite_subtiles([{
        'min': 0,
        'max': 1,
        'grid': (0, 0),
        'image': 255 * np.ones((1024, 1024), dtype=np.uint8),
        'color': color_orange
    }], (1024, 1024), (0, 0), (1024, 1024))

    np.testing.assert_allclose(expected, result)


def test_composite_subtiles_real(real_tiles_green_mask, real_tiles_red_mask,
                                 real_stitched_with_gamma):
    '''Ensure 1024 x 1024 image matches image rendered without tiling.'''

    expected = real_stitched_with_gamma

    inputs = []

    for y in range(0, 4):
        for x in range(0, 4):
            inputs += [{
                'min': 0.006,
                'max': 0.024,
                'grid': (y, x),
                'image': real_tiles_green_mask[y][x],
                'color': color_green
            }, {
                'min': 0,
                'max': 1,
                'grid': (y, x),
                'image': real_tiles_red_mask[y][x],
                'color': color_red
            }]

    result = composite_subtiles(inputs, (256, 256),
                                (0, 0), (1024, 1024))

    np.testing.assert_allclose(expected, np.uint8(255*result))
