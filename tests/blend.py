'''Compare blend results with expected output'''

import pytest
import numpy as np
from minerva_lib.blend import make_rgb
from minerva_lib.blend import clip_image
from minerva_lib.blend import linear_rgb


@pytest.fixture
def full_u8():
    return int(255)


@pytest.fixture
def full_u16():
    return int(65535)


@pytest.fixture
def range_all():
    return np.array([0, 1], dtype=np.float32)


@pytest.fixture
def range_high():
    return np.array([0.5, 1], dtype=np.float32)


@pytest.fixture
def range_low(full_u8, full_u16):
    return np.array([0, full_u8 / full_u16], dtype=np.float32)


@pytest.fixture(params=['range_all', 'range_high', 'range_low'])
def ranges(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def color_white():
    return np.array([1, 1, 1], dtype=np.float32)


@pytest.fixture
def color_yellow():
    return np.array([1, 1, 0], dtype=np.float32)


@pytest.fixture
def color_green():
    return np.array([0, 1, 0], dtype=np.float32)


@pytest.fixture
def color_blue():
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture
def color_red():
    return np.array([1, 0, 0], dtype=np.float32)


@pytest.fixture
def color_khaki(full_u8):
    return np.array([240, 230, 140], dtype=np.float32) / full_u8


@pytest.fixture(params=['color_white', 'color_yellow', 'color_green',
                        'color_blue', 'color_red', 'color_khaki'])
def colors(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def u16_low_med_high(full_u8, full_u16):
    return np.array([[0], [full_u8], [full_u16]], dtype=np.uint16)


@pytest.fixture
def f32_low_med_high(full_u8, full_u16):
    return np.array([[0], [full_u8 / full_u16], [1]], dtype=np.float32)


@pytest.fixture
def channel_check(full_u16):
    return np.array([
        [0, full_u16],
        [full_u16, 0]
    ], dtype=np.uint16)


@pytest.fixture
def channel_check_inverse(full_u16):
    return np.array([
        [full_u16, 0],
        [0, full_u16]
    ], dtype=np.uint16)


def test_clip_image_range_all(u16_low_med_high, color_white, range_all):
    '''Extract high values from image in channel dictionary'''

    expected = (
        np.array([[0], [255 / 65535], [1]], dtype=np.float64),
        np.array([1, 1, 1], dtype=np.float64)
    )

    result = clip_image({
        'image': u16_low_med_high,
        'color': color_white,
        'min': range_all[0],
        'max': range_all[1]
    })

    np.testing.assert_array_equal(expected[0], result[0])
    np.testing.assert_array_equal(expected[1], result[1])


def test_clip_image_range_high(u16_low_med_high, color_white, range_high):
    '''Extract high values from image in channel dictionary'''

    expected = (
        np.array([[0], [0], [1]], dtype=np.float64),
        np.array([1, 1, 1], dtype=np.float64)
    )

    result = clip_image({
        'image': u16_low_med_high,
        'color': color_white,
        'min': range_high[0],
        'max': range_high[1]
    })

    np.testing.assert_array_equal(expected[0], result[0])
    np.testing.assert_array_equal(expected[1], result[1])


def test_clip_image_range_low(u16_low_med_high, color_white, range_low):
    '''Extract low values from image in channel dictionary'''

    expected = (
        np.array([[0], [1], [1]], dtype=np.float64),
        np.array([1, 1, 1], dtype=np.float64)
    )

    result = clip_image({
        'image': u16_low_med_high,
        'color': color_white,
        'min': range_low[0],
        'max': range_low[1]
    })

    np.testing.assert_array_equal(expected[0], result[0])
    np.testing.assert_array_equal(expected[1], result[1])


def test_make_rgb_color_red(f32_low_med_high, color_red):
    '''Blend an image with one channel, testing red color'''

    expected = np.array([
        [[0, 0, 0]],
        [[255 / 65535, 0, 0]],
        [[1, 0, 0]]
    ], dtype=np.float32)

    result = make_rgb(f32_low_med_high, color_red)

    np.testing.assert_array_equal(expected[:, :, 0], next(result))
    np.testing.assert_array_equal(expected[:, :, 1], next(result))
    np.testing.assert_array_equal(expected[:, :, 2], next(result))


def test_make_rgb_color_white(f32_low_med_high, range_all, color_white):
    '''Blend an image with one channel, testing white color'''

    expected = np.array([
        [[0, 0, 0]],
        [color_white * 255 / 65535],
        [color_white]
    ], dtype=np.float32)

    result = make_rgb(f32_low_med_high, color_white)

    np.testing.assert_array_equal(expected[:, :, 0], next(result))
    np.testing.assert_array_equal(expected[:, :, 1], next(result))
    np.testing.assert_array_equal(expected[:, :, 2], next(result))


def test_make_rgb_color_khaki(f32_low_med_high, color_khaki):
    '''Make an image with one channel, testing khaki color
    Colors of any lightness/chroma should correctly normalize
    '''

    expected = np.array([
        [[0, 0, 0]],
        [color_khaki * 255 / 65535],
        [color_khaki]
    ], dtype=np.float32)

    result = make_rgb(f32_low_med_high, color_khaki)

    np.testing.assert_array_equal(expected[:, :, 0], next(result))
    np.testing.assert_array_equal(expected[:, :, 1], next(result))
    np.testing.assert_array_equal(expected[:, :, 2], next(result))


def test_linear_rgb_khaki_low(u16_low_med_high, range_low, color_khaki):
    '''Blend an image with one channel, testing khaki at low range
    Colors of any lightness/chroma should set all over max to 1
    '''

    expected = np.array([
        [[0, 0, 0]],
        [color_khaki],
        [color_khaki],
    ], dtype=np.float32) ** (1 / 2.2)

    result = linear_rgb([{
        'image': u16_low_med_high,
        'color': color_khaki,
        'min': range_low[0],
        'max': range_low[1]
    }])

    np.testing.assert_array_equal(expected, result)


def test_linear_rgb_two_channel(channel_check, channel_check_inverse,
                                range_all, color_blue, color_yellow):
    '''Test blending an image with two channels'''

    expected = np.array([
        [color_yellow, color_blue],
        [color_blue, color_yellow],
    ], dtype=np.float32)

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


def test_linear_rgb_size_mismatch(range_all, color_white):
    '''Test supplying channels with different dimensions'''

    input_channels = [
        {
            'image': np.array([[0, 0, 0]], dtype=np.uint16),
            'color': color_white,
            'min': range_all[0],
            'max': range_all[1]
        },
        {
            'image': np.array([[0, 65535]], dtype=np.uint16),
            'color': color_white,
            'min': range_all[0],
            'max': range_all[1]
        }
    ]

    with pytest.raises(ValueError,
                       match=r'.*broadcast.*'):
        linear_rgb(input_channels)


def test_linear_rgb_size_zero():
    '''Test supplying no channels'''

    with pytest.raises(ValueError,
                       match=r'At least one channel must be specified'):
        linear_rgb([])
