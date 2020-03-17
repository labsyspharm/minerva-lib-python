'''Compare blend results with expected output'''

import pytest
import numpy as np
from minerva_lib.render import composite_channel, composite_channels
import time, random, math

@pytest.fixture
def range_all():
    return np.array([0, 1], dtype=np.float32)


@pytest.fixture
def range_high():
    return np.array([0.5, 1], dtype=np.float32)


@pytest.fixture
def range_low():
    return np.array([0, 12345 / 65535], dtype=np.float32)


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
def color_khaki():
    return np.array([240, 230, 140], dtype=np.float32) / 255


@pytest.fixture(params=['color_white', 'color_yellow', 'color_green',
                        'color_blue', 'color_red', 'color_khaki'])
def colors(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def u16_3value_channel():
    return np.array([[0], [12345], [65535]], dtype=np.uint16)


@pytest.fixture
def f32_3value_rgb_buffer():
    return np.zeros((3, 1, 3), dtype=np.float32)


@pytest.fixture
def u16_checkered_channel():
    return np.array([
        [0, 65535],
        [65535, 0]
    ], dtype=np.uint16)


@pytest.fixture
def u16_checkered_channel_inverse():
    return np.array([
        [65535, 0],
        [0, 65535]
    ], dtype=np.uint16)


def test_channel_range_high(u16_3value_channel, color_white, range_high,
                            f32_3value_rgb_buffer):
    '''Extract high values from image in channel dictionary'''

    expected = np.array([
        [[0, 0, 0]],
        [[0, 0, 0]],
        [color_white]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_white, *range_high)

    np.testing.assert_allclose(expected, result)


def test_channel_range_low(u16_3value_channel, color_white, range_low,
                           f32_3value_rgb_buffer):
    '''Extract low values from image in channel dictionary'''

    expected = np.array([
        [[0, 0, 0]],
        [color_white],
        [color_white]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_white, *range_low)

    np.testing.assert_allclose(expected, result)


def test_channel_color_red(u16_3value_channel, color_red, range_all,
                           f32_3value_rgb_buffer):
    '''Blend an image with one channel, testing red color'''

    expected = np.array([
        [[0, 0, 0]],
        [[12345 / 65535, 0, 0]],
        [[1, 0, 0]]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_red, *range_all)

    np.testing.assert_allclose(expected, result)


def test_channel_color_white(u16_3value_channel, color_white, range_all,
                             f32_3value_rgb_buffer):
    '''Blend an image with one channel, testing white color'''

    expected = np.array([
        [[0, 0, 0]],
        [color_white * 12345 / 65535],
        [color_white]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_white, *range_all)

    np.testing.assert_allclose(expected, result)


def test_channel_target_is_out(u16_3value_channel, color_white, range_all,
                               f32_3value_rgb_buffer):
    '''Blend an image in place by providing an output argument'''

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_white, *range_all,
                               out=f32_3value_rgb_buffer)

    assert f32_3value_rgb_buffer is result


def test_channel_color_khaki(u16_3value_channel, color_khaki, range_all,
                             f32_3value_rgb_buffer):
    '''Make an image with one channel, testing khaki color
    Ensure any color mappings normalize between 0 and 1
    '''

    expected = np.array([
        [[0, 0, 0]],
        [color_khaki * 12345 / 65535],
        [color_khaki]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_khaki, *range_all)

    np.testing.assert_allclose(expected, result)


def test_channel_khaki_low(u16_3value_channel, color_khaki, range_low,
                           f32_3value_rgb_buffer):
    '''Blend an image with one channel, testing khaki at low range
    Ensure overly bright values are clipped to 1
    '''

    expected = np.array([
        [[0, 0, 0]],
        [color_khaki],
        [color_khaki],
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_khaki, *range_low)

    np.testing.assert_allclose(expected, result)


def test_channels_two_channel(u16_checkered_channel,
                              u16_checkered_channel_inverse,
                              color_blue, color_yellow, range_all):
    '''Test blending an image with two channels'''

    expected = np.array([
        [color_yellow, color_blue],
        [color_blue, color_yellow],
    ], dtype=np.float32)

    result = composite_channels([
        {
            'image': u16_checkered_channel,
            'color': color_blue,
            'min': range_all[0],
            'max': range_all[1]
        },
        {
            'image': u16_checkered_channel_inverse,
            'color': color_yellow,
            'min': range_all[0],
            'max': range_all[1]
        }
    ])

    np.testing.assert_allclose(expected, result)


def test_channels_size_mismatch(color_white, range_all):
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

    with pytest.raises(ValueError):
        composite_channels(input_channels)


def test_channels_size_zero():
    '''Test supplying no channels'''

    with pytest.raises(ValueError):
        composite_channels([])

def _blend_random_image(times, output):
    total = 0
    for i in range(times):
        channels = []
        for channel in range(4):
            test_img = np.random.random_integers(0, 255, (1024, 1024))
            color = (random.random(), random.random(), random.random())
            a = random.random()
            b = random.random()

            channel = {
                "image": test_img,
                "color": color,
                "min": min(a,b),
                "max": max(a,b)
            }
            channels.append(channel)

        start = time.time()
        composite_channels(channels)
        t = round((time.time() - start) * 1000)
        total += t

    return total

def test_performance():
    print("Initializing test image arrays")
    shape = (1024, 1024)
    # Shape of 3 color image
    shape_color = shape + (3,)
    output = np.zeros(shape_color, dtype=np.float32)
    total = 0

    # warm up
    _blend_random_image(2, output)

    random.seed(2020)

    # actual benchmark
    total += _blend_random_image(20, output)

    print("Total time: ", total)
    print("Per composite: ", total // 20)
