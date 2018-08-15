'''Compare crop results with expected output'''

import pytest
import numpy as np
from minerva_lib.crop import scale_image_nearest_neighbor


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
