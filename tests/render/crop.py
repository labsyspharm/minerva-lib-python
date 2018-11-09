'''Compare crop results with expected output.'''

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
    '''One 4x4 pixel image of four identical 2x2 tiles.'''

    return np.array([
        [[0, 0, 0], [1, 1, 1]] * 2,
        [[1, 1, 1], [0, 0, 0]] * 2,
    ] * 2)


def test_scale_image_aliasing(checker_4x4):
    '''Ensure expected nearest neighbor aliasing when scaling in y and x.

    The function does not interpolate, so the checkerboard will not evenly
    sample. The second column and second row will be missing, so the other
    values will shift to scale from 4x4 to 3x3.
    '''

    expected = np.array([
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
    ])

    result = scale_image_nearest_neighbor(checker_4x4, 3 / 4)

    np.testing.assert_allclose(expected, result)


def test_scale_image_asymetry(checker_4x4):
    '''Ensure expected nearest neighbor aliasing when scaling only in y.

    Due to aliasing, the second column will be missing, so the other columns
    will shift to scale from 4x4 to 3x4. All four rows should remain with 3
    of the original 4 values.
    '''

    expected = np.array([
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
    ])

    result = scale_image_nearest_neighbor(checker_4x4, (1, 3 / 4))

    np.testing.assert_allclose(expected, result)


def test_scale_image_invalid_factor():
    '''Ensure inability to downsample arbitrary color image to 0%.'''

    with pytest.raises(ValueError):
        scale_image_nearest_neighbor(np.array([
                                        [[0, 0, 0], [0, 0, 0]],
                                        [[0, 0, 0], [0, 0, 0]]
                                    ]), (0, 0))


def test_get_optimum_pyramid_level_higher():
    '''Test higher resolution than needed for output shape.

    We would downscale a 6x6 image to 4x4 pixels if we prefer an output
    image taken from a higher-resolution pyramid level. For a 6x6 image,
    this means we need the highest resolution available, level 0.
    '''

    expected = 0

    result = get_optimum_pyramid_level((6, 6), 2,
                                       4, True)

    assert expected == result


def test_get_optimum_pyramid_level_lower():
    '''Test lower resolution than needed for output shape.

    We would upscale a 3x3 image to 4x4 pixels if we prefer an output
    image taken from a lower-resolution pyramid level. For a 6x6 image,
    this means we need a half-resolution 3x3 image, level 1.
    '''

    expected = 1

    result = get_optimum_pyramid_level((6, 6), 2,
                                       4, False)

    assert expected == result


def test_transform_coordinates_full_scale():
    '''Ensure scaling to level 0 keeps the coordinates unchanged.'''

    expected = np.array((6, 6), dtype=np.int64)

    result = transform_coordinates_to_level((6, 6), 0)

    np.testing.assert_array_equal(expected, result)


def test_transform_coordinates_half_scale():
    '''Ensure scaling to level 1 divides the coordinates in half.'''

    expected = np.array((3, 3), dtype=np.int64)

    result = transform_coordinates_to_level((6, 6), 1)

    np.testing.assert_array_equal(expected, result)


def test_select_position_inner_tile():
    '''Test ability to position tile within the output image.

    The origin of a 2x2 tile at column 1, row 1 of the tile grid
    should be 2, 2 within any output image with an origin at 0, 0.
    '''

    expected = (2, 2)

    result = select_position((1, 1), (2, 2), (0, 0))

    np.testing.assert_array_equal(expected, result)


def test_get_first_grid_inner_tile():
    '''Test ability to index the output image origin within the tile grid.

    The 2x2 tile found at column 1, row 1 of the tile grid should
    begin at the origin of any output image with an origin at 2, 2.
    '''

    expected = (1, 1)

    result = get_region_first_grid((2, 2), (2, 2))

    np.testing.assert_array_equal(expected, result)


def test_get_grid_shape_inner_tiles():
    '''Ensure the grid shape counts inner tiles and partial edge tiles.

    When we measure 6x6 pixels starting from an origin of 3x3 on a grid of
    2x2 tiles, the region should cover 4 columns and 4 rows of tiles. The
    region begins within the tile at column 1, row 1. The region ends in
    the tile at column 4, row 4. We must count the full range of 4 tiles
    in either dimension.
    '''

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


def test_select_grids_odd_edges():
    '''Ensure selection of tiles identifies partial tiles at edges.

    The output image needs exactly 1 pixel from each of 4 tiles along
    two rows and columns of the tile grid. This should give the correct
    indices of all partially needed tiles.
    '''

    expected = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2)
    ]

    result = select_grids((2, 2), (3, 3), (2, 2))

    np.testing.assert_array_equal(expected, result)


def test_select_grids_odd_end():
    '''Ensure selection of tiles includes a full tile and partial tiles.

    The output image needs all pixels from the first tile as well as
    portions of the other three tiles on the edge of the region. This
    should give the correct indices of the full and partial tiles.
    '''

    expected = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    result = select_grids((2, 2), (0, 0), (3, 3))

    np.testing.assert_array_equal(expected, result)


def test_select_subregion_odd_edges():
    '''Test tile subregion for tile containing the output image origin.

    The output image needs only the last pixel from the tile that contains
    the output image origin in y and x. The returned coordinates should
    specify the selection of the last pixel from this tile in y and x.
    '''

    expected = [
        (1, 1),
        (2, 2)
    ]

    result = select_subregion((0, 0), (2, 2),
                              (1, 1), (2, 2))

    np.testing.assert_array_equal(expected, result)


def test_select_subregion_odd_end():
    '''Test tile subregion for tile at end of output image.

    The output image needs only the first pixel from the tile that contains
    the end of the output image in y and x. The returned coordinates should
    specify the selection of the first pixel from this tile in y and x.
    '''

    expected = [
        (0, 0),
        (1, 1)
    ]

    result = select_subregion((1, 1), (2, 2),
                              (0, 0), (3, 3))

    np.testing.assert_array_equal(expected, result)


def test_extract_subtile_half_tile():
    '''Test extracting a subtile at half the height of a full tile.

    The output image requires the last row of pixels from a tile in row 0 and
    the first row of pixels from a tile in row 1. When cropping from row 0,
    the result should match the last row of pixels in the tile.
    '''

    expected = np.array([
        [[1, 1, 1], [0, 0, 0]]
    ])

    result = extract_subtile((0, 0), (2, 2),
                             (1, 0), (2, 2),
                             np.array([
                                [[0, 0, 0], [1, 0, 0]],
                                [[1, 1, 1], [0, 0, 0]]
                             ]))

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_normalize():
    '''Test uniform normalization of 16-bit and 8-bit integers.

    The two function calls should each add normalized pixels from tiles
    of completely different bit depths.
    '''

    expected = np.array([
        [[1, 1, 1], [1, 1, 1]]
    ])

    result = composite_subtile(np.zeros((1, 2) + (3,)),
                               np.array([[255, 0]], dtype=np.uint8),
                               (0, 0), color_white, 0, 1)
    result = composite_subtile(result,
                               np.array([[0, 65535]], dtype=np.uint16),
                               (0, 0), color_white, 0, 1)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_threshold():
    '''Test addition of low-threshold and high-threshold image pixels.

    The first function call adds dark red to both pixels, then the second
    function call adds gray to the second pixel. The second pixel should be
    a light pink due to the addition of the two colors.
    '''
    expected = np.array([
        [[0.5, 0, 0], [1, 0.5, 0.5]]
    ])

    result = composite_subtile(np.zeros((1, 2) + (3,)),
                               np.array([[55535] * 2], dtype=np.uint16),
                               (0, 0), color_red, 45535 / 65535, 1)
    result = composite_subtile(result,
                               np.array([[0, 50]], dtype=np.uint16),
                               (0, 0), color_white, 0, 100 / 65535)

    np.testing.assert_array_equal(expected, result)


def test_composite_subtile_blend():
    '''Test linearly blending red and green for a single tile.

    The first function call should add red to the last row of pixels.
    The last function call should add green to the last column of pixels.
    The last pixel in x nad y should show the composite sum as yellow.
    '''

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


def test_composite_subtile_stitch():
    '''Test correct alignment when compositing two tiles at two y indices.

    The first function call should add to the second pixels in the first row
    and column of pixels while the second function call should add to the
    second pixel in the last row of pixels.
    '''

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


def test_composite_subtiles_nonsquare():
    '''Stitch a non-square image from 1 full and 3 edge tiles.

    The four tiles should completely cover a 1080x1920 output image. The
    tiles must line up such that the dimensions of each differently sized
    tile exactly fits at the right position to fill the full rectangular
    image when combined.
    '''

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


def test_composite_subtiles_gamma():
    '''Render a single tile testing gamma correction.

    The color orange, when fully rendered with a gamma of 2.2, should
    show a gamma-corrected higher value in the green channel with no change
    at the higher and lower extremes of red and blue.
    '''

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
    '''Ensure sixteen tiles match the image made with one large tile.

    Sixteen tiles per channel should stitch and composite to exactly match
    the same image previously rendered with one tile per channel.
    '''

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
