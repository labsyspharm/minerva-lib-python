'''Compare blend results with expected output'''

import pytest
import numpy as np
from minerva_lib.blend import to_f32
from minerva_lib.blend import f32_to_bgr
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
    return np.float32([0, 1, 1])


@pytest.fixture
def color_green():
    return np.float32([0, 1, 0])


@pytest.fixture
def color_blue():
    return np.float32([1, 0, 0])


@pytest.fixture
def color_red():
    return np.float32([0, 0, 1])


@pytest.fixture
def color_khaki():
    return np.float32([140, 230, 240]) / 255.0


@pytest.fixture(params=['color_white', 'color_yellow', 'color_green',
                        'color_blue', 'color_red', 'color_khaki'])
def colors(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def f32_channel_low_med_high():
    return np.float32([
        [0.0],
        [256.0 / 65535.0],
        [1.0],
    ])


@pytest.fixture
def channel_low_med_high():
    return np.uint16([[0], [256], [65535]])


@pytest.fixture
def channel_check():
    return np.uint16([
        [0, 65535],
        [65535, 0]
    ])


@pytest.fixture
def channel_check_inverse():
    return np.uint16([
        [65535, 0],
        [0, 65535]
    ])


def test_range_all(channel_low_med_high, color_white, range_all):
    '''Blend an image with one channel, testing full range'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_range_high(channel_low_med_high, color_white, range_high):
    '''Blend an image with one channel, testing high range'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[255, 255, 255]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_high[0],
        'max': range_high[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_range_low(channel_low_med_high, color_white, range_low):
    '''Blend an image with one channel, testing low range'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[255, 255, 255]],
        [[0, 0, 0]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_low[0],
        'max': range_low[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_white(channel_low_med_high, range_all, color_white):
    '''Blend an image with one channel, testing white color'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_white,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_red(channel_low_med_high, range_all, color_red):
    '''Blend an image with one channel, testing red color'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 0, 1]],
        [[0, 0, 255]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_red,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_khaki(channel_low_med_high, range_all, color_khaki):
    '''Blend an image with one channel, testing khaki color
    Colors of any lightness/chroma should map low uint16 input values to 1
    '''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[140, 230, 240]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_khaki,
        'min': range_all[0],
        'max': range_all[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_color_khaki_range_low(channel_low_med_high, range_low, color_khaki):
    '''Blend an image with one channel, testing khaki at low range
    Colors of any lightness/chroma should map inputs above threshhold to 0
    '''

    expected = np.uint8([
        [[0, 0, 0]],
        [[140, 230, 240]],
        [[0, 0, 0]]
    ])

    result = linear_bgr([{
        'image': channel_low_med_high,
        'color': color_khaki,
        'min': range_low[0],
        'max': range_low[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_multi_channel(channel_check, channel_check_inverse, range_all,
                       color_blue, color_yellow):
    '''Test blending an image with multiple channels'''

    expected = 128 * np.uint8([
        [color_yellow, color_blue],
        [color_blue, color_yellow],
    ])

    result = linear_bgr([
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
            'image': np.uint16([0]),
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
                       match=r'All channel images must have equal dimensions'):
        linear_bgr(input_channels)


def test_channel_size_zero():
    '''Test supplying no channels'''

    with pytest.raises(ValueError,
                       match=r'At least one channel must be specified'):
        linear_bgr([])


def test_to_f32_full(channel_low_med_high, f32_channel_low_med_high):
    ''' Test conversion to f32 across uint16 range'''

    expected = f32_channel_low_med_high
    result = to_f32(channel_low_med_high)

    np.testing.assert_array_equal(expected, result)


def test_to_f32_float_input(f32_channel_low_med_high):
    '''Test supplying floating points when unsigned integers are expected'''

    with pytest.raises(ValueError,
                       match=r'Scaling to 0,1 requires unsigned integers'):
        to_f32(f32_channel_low_med_high)


def test_f32_to_bgr_white(channel_low_med_high, f32_channel_low_med_high):
    ''' Test conversion from f32 to black, gray, white'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]]
    ])

    result = f32_to_bgr(f32_channel_low_med_high)

    np.testing.assert_array_equal(expected, result)


def test_f32_to_bgr_yellow(color_yellow, channel_low_med_high,
                           f32_channel_low_med_high):
    ''' Test conversion from f32 to yellow gradient'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 1, 1]],
        [[0, 255, 255]]
    ])

    result = f32_to_bgr(f32_channel_low_med_high, color_yellow)

    np.testing.assert_array_equal(expected, result)


def test_f32_to_bgr_int_input(channel_low_med_high):
    '''Test supplying floating points when unsigned integers are expected'''

    with pytest.raises(ValueError,
                       match=r'Color image requires values from 0,1'):
        f32_to_bgr(channel_low_med_high)
