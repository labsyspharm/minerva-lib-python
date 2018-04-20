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
def f32_image_1channel():
    return np.float32([
        [0.0],
        [256.0 / 65536.0],
        [65535.0 / 65536.0],
    ])


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


def test_range_all(image_1channel, color_white, range_all):
    '''Blend an image with one channel, testing full range'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_white],
                        ranges=[range_all])

    np.testing.assert_array_equal(expected, result)


def test_range_high(image_1channel, color_white, range_high):
    '''Blend an image with one channel, testing high range'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[255, 255, 255]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_white],
                        ranges=[range_high])

    np.testing.assert_array_equal(expected, result)


def test_range_low(image_1channel, color_white, range_low):
    '''Blend an image with one channel, testing low range'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[255, 255, 255]],
        [[0, 0, 0]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_white],
                        ranges=[range_low])

    np.testing.assert_array_equal(expected, result)


def test_color_white(image_1channel, range_all, color_white):
    '''Blend an image with one channel, testing white color'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_white],
                        ranges=[range_all])

    np.testing.assert_array_equal(expected, result)


def test_color_red(image_1channel, range_all, color_red):
    '''Blend an image with one channel, testing red color'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 0, 1]],
        [[0, 0, 255]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_red],
                        ranges=[range_all])

    np.testing.assert_array_equal(expected, result)


def test_color_khaki(image_1channel, range_all, color_khaki):
    '''Blend an image with one channel, testing khaki color
    Colors of any lightness/chroma should map low uint16 input values to 0
    '''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[140, 230, 240]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_khaki],
                        ranges=[range_all])

    np.testing.assert_array_equal(expected, result)


def test_color_khaki_range_low(image_1channel, range_low, color_khaki):
    '''Blend an image with one channel, testing khaki at low range
    Colors of any lightness/chroma should map inputs above threshhold to 0
    '''

    expected = np.uint8([
        [[0, 0, 0]],
        [[140, 230, 240]],
        [[0, 0, 0]]
    ])

    result = linear_bgr(image_1channel,
                        colors=[color_khaki],
                        ranges=[range_low])

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


def test_to_f32_full(image_1channel, f32_image_1channel):
    ''' Test conversion to f32 across uint16 range'''

    expected = f32_image_1channel
    result = to_f32(image_1channel[0])

    np.testing.assert_array_equal(expected, result)


def test_to_f32_float_input(f32_image_1channel):
    '''Test supplying floating points when unsigned integers are expected'''

    with pytest.raises(ValueError,
                       match=r'Scaling to 0,1 requires unsigned integers'):
        to_f32(f32_image_1channel)


def test_f32_to_bgr_white(image_1channel, f32_image_1channel):
    ''' Test conversion from f32 to black, gray, white'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]]
    ])

    result = f32_to_bgr(f32_image_1channel)

    np.testing.assert_array_equal(expected, result)


def test_f32_to_bgr_yellow(color_yellow, image_1channel, f32_image_1channel):
    ''' Test conversion from f32 to yellow gradient'''

    expected = np.uint8([
        [[0, 0, 0]],
        [[0, 1, 1]],
        [[0, 255, 255]]
    ])

    result = f32_to_bgr(f32_image_1channel, color_yellow)

    np.testing.assert_array_equal(expected, result)


def test_f32_to_bgr_int_input(image_1channel):
    '''Test supplying floating points when unsigned integers are expected'''

    with pytest.raises(ValueError,
                       match=r'Color image requires values from 0,1'):
        f32_to_bgr(image_1channel[0])
