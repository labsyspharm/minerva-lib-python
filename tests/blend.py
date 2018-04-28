'''Compare blend results with expected output'''

import pytest
import numpy as np
from minerva_lib.blend import handle_channel
from minerva_lib.blend import linear_rgb


@pytest.fixture
def full_u8():
    return int(255)


@pytest.fixture
def full_u16():
    return int(65535)


@pytest.fixture
def range_all():
    return np.float32([0, 1])


@pytest.fixture
def range_high():
    return np.float32([0.5, 1])


@pytest.fixture
def range_low(full_u8, full_u16):
    return np.float32([0, full_u8 / full_u16])


@pytest.fixture(params=['range_all', 'range_high', 'range_low'])
def ranges(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def color_white():
    return np.float32([1, 1, 1])


@pytest.fixture
def color_yellow():
    return np.float32([1, 1, 0])


@pytest.fixture
def color_green():
    return np.float32([0, 1, 0])


@pytest.fixture
def color_blue():
    return np.float32([0, 0, 1])


@pytest.fixture
def color_red():
    return np.float32([1, 0, 0])


@pytest.fixture
def color_khaki(full_u8):
    return np.float32([240, 230, 140]) / full_u8


@pytest.fixture(params=['color_white', 'color_yellow', 'color_green',
                        'color_blue', 'color_red', 'color_khaki'])
def colors(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def channel_low_med_high(full_u8, full_u16):
    return np.uint16([[0], [full_u8], [full_u16]])


@pytest.fixture
def channel_check(full_u16):
    return np.uint16([
        [0, full_u16],
        [full_u16, 0]
    ])


@pytest.fixture
def channel_check_inverse(full_u16):
    return np.uint16([
        [full_u16, 0],
        [0, full_u16]
    ])


def test_handle_channel(channel_low_med_high, color_white, range_high):
    '''Extract scaled color and threshholded image from channel dictionary'''

    expected = (
        np.float32([[0], [0], [1]]),
        np.float32([1, 1, 1])
    )

    result = handle_channel({
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_high[0],
        'max': range_high[1]
    })

    np.testing.assert_array_equal(expected[0], result[0])
    np.testing.assert_array_equal(expected[1], result[1])


def test_range_high(channel_low_med_high, color_white, range_high):
    '''Blend an image with one channel, testing high range'''

    expected = np.float32([
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[1, 1, 1]]
    ])

    result = linear_rgb([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_high[0],
        'max': range_high[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_range_low(channel_low_med_high, color_white, range_low):
    '''Blend an image with one channel, testing low range'''

    expected = np.float32([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[1, 1, 1]]
    ])

    result = linear_rgb([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_low[0],
        'max': range_low[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_white(channel_low_med_high, range_all, color_white):
    '''Blend an image with one channel, testing white color'''

    expected = np.float32([
        [[0, 0, 0]],
        [[255, 255, 255]],
        [[65535, 65535, 65535]]
    ]) / 65535

    result = linear_rgb([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_red(channel_low_med_high, range_all, color_red):
    '''Blend an image with one channel, testing red color'''

    expected = np.float32([
        [[0, 0, 0]],
        [[255, 0, 0]],
        [[65535, 0, 0]]
    ]) / 65535

    result = linear_rgb([{
        'image': channel_low_med_high,
        'color': color_red,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_khaki(channel_low_med_high, range_all, color_khaki):
    '''Blend an image with one channel, testing khaki color
    Colors of any lightness/chroma should correctly normalize
    '''

    expected = np.float32([
        [[0, 0, 0]],
        [color_khaki * 255],
        [color_khaki * 65535]
    ]) / 65535

    result = linear_rgb([{
        'image': channel_low_med_high,
        'color': color_khaki,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_khaki_range_low(channel_low_med_high, range_low, color_khaki):
    '''Blend an image with one channel, testing khaki at low range
    Colors of any lightness/chroma should set all over max to 1
    '''

    expected = np.float32([
        [[0, 0, 0]],
        [color_khaki],
        [color_khaki],
    ])

    result = linear_rgb([{
        'image': channel_low_med_high,
        'color': color_khaki,
        'min': range_low[0],
        'max': range_low[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_multi_channel(channel_check, channel_check_inverse, range_all,
                       color_blue, color_yellow):
    '''Test blending an image with multiple channels'''

    expected = 1.0 * np.float32([
        [color_yellow, color_blue],
        [color_blue, color_yellow],
    ])

    result = linear_rgb([
        {
            'image': channel_check,
            'color': color_blue,
            'min': range_all[0],
            'max': range_all[1]
        },
        {
            'image': channel_check_inverse,
            'color': color_yellow,
            'min': range_all[0],
            'max': range_all[1]
        }
    ])

    np.testing.assert_array_equal(expected, result)


def test_channel_size_mismatch(range_all, color_white):
    '''Test supplying channels with different dimensions'''

    input_channels = [
        {
            'image': np.uint16([[0, 0, 0]]),
            'color': color_white,
            'min': range_all[0],
            'max': range_all[1]
        },
        {
            'image': np.uint16([[0, 65535]]),
            'color': color_white,
            'min': range_all[0],
            'max': range_all[1]
        }
    ]

    with pytest.raises(ValueError,
                       match=r'.*broadcast.*'):
        linear_rgb(input_channels)


def test_channel_size_zero():
    '''Test supplying no channels'''

    with pytest.raises(ValueError,
                       match=r'At least one channel must be specified'):
        linear_rgb([])
