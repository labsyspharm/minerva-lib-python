'''Compare blend results with expected output'''

import pytest
import numpy as np
from minerva_lib.blend import linear_bgr


@pytest.fixture
def range_all():
    return np.float32([0, 1])


@pytest.fixture
def range_high():
    return np.float32([0.5, 1])


@pytest.fixture
def range_low():
    return np.float32([0, 256. / 65535.])


@pytest.fixture(params=['range_all', 'range_high', 'range_low'])
def ranges(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def color_white():
    return np.float32([1, 1, 1])


@pytest.fixture
def color_yellow():
    return np.float32([1, 1, 1])


@pytest.fixture
def color_green():
    return np.float32([1, 1, 1])


@pytest.fixture
def color_blue():
    return np.float32([1, 1, 1])


@pytest.fixture
def color_red():
    return np.float32([1, 1, 1])


@pytest.fixture(params=['color_white', 'color_yellow', 'color_green',
                        'color_blue', 'color_red'])
def colors(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def image_1channel():
    return np.uint16([[[0], [256], [65535]]])


@pytest.fixture
def image_2channel():
    return np.uint16([[
        [0, 65535],
        [65535, 0],
    ], [
        [65535, 0],
        [0, 65535],
    ]])


@pytest.mark.parametrize('rngs,expected', [
    (
        np.float32([0, 1]),
        np.uint8([[[0, 0, 0]], [[1, 1, 1]], [[255, 255, 255]]])
    ),
    (
        np.float32([0, 256. / 65535.]),
        np.uint8([[[0, 0, 0]], [[255, 255, 255]], [[0, 0, 0]]])
    ),
    (
        np.float32([0.5, 1]),
        np.uint8([[[0, 0, 0]], [[0, 0, 0]], [[255, 255, 255]]])
    )
])
def test_range(image_1channel, color_white, rngs, expected):
    '''Blend an image with one channel, testing ranges'''

    result = linear_bgr(image_1channel,
                        colors=[color_white],
                        ranges=[rngs])

    np.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize('colors,expected', [
    (
        np.float32([1, 1, 1]),
        np.uint8([[[0, 0, 0]], [[1, 1, 1]], [[255, 255, 255]]])
    ),
    (
        np.float32([0, 1, 1]),
        np.uint8([[[0, 0, 0]], [[0, 1, 1]], [[0, 255, 255]]])
    ),
    (
        np.float32([0, 1, 0]),
        np.uint8([[[0, 0, 0]], [[0, 1, 0]], [[0, 255, 0]]])
    ),
    (
        np.float32([1, 0, 0]),
        np.uint8([[[0, 0, 0]], [[1, 0, 0]], [[255, 0, 0]]])
    ),
    (
        np.float32([0, 0, 1]),
        np.uint8([[[0, 0, 0]], [[0, 0, 1]], [[0, 0, 255]]])
    )
])
def test_color(image_1channel, range_all, colors, expected):
    '''Blend an image with one channel, testing colors'''

    result = linear_bgr(image_1channel,
                        colors=[colors],
                        ranges=[range_all])

    np.testing.assert_array_equal(expected, result)


def test_multi_channel(image_2channel, range_all, color_blue, color_yellow):
    '''Test blending an image with multiple channels'''

    expected = 127 * np.uint8([
        [color_yellow, color_blue],
        [color_blue, color_yellow],
    ])

    result = linear_bgr(image_2channel,
                        colors=[color_blue, color_yellow],
                        ranges=[range_all, range_all])

    np.testing.assert_array_equal(expected, result)


def test_channel_range_mismatch(image_2channel, range_all, color_white):
    '''Test supplying wrong number of ranges for the number of channels'''

    with pytest.raises(ValueError,
                       match=r'One range per channel must be specified'):
        linear_bgr(image_2channel,
                   colors=[color_white, color_white],
                   ranges=[range_all])


def test_channel_color_mismatch(image_2channel, range_all, color_white):
    '''Test supplying wrong number of colors for the number of channels'''

    with pytest.raises(ValueError,
                       match=r'One color per channel must be specified'):
        linear_bgr(image_2channel,
                   colors=[color_white],
                   ranges=[range_all, range_all])
