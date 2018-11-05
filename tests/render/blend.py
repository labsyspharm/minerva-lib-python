'''Compare blend results with expected output'''

import pytest
import numpy as np
from minerva_lib.render import composite_channel, composite_channels


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
def color_blue():
    return np.array([0, 0, 1], dtype=np.float32)


@pytest.fixture
def color_red():
    return np.array([1, 0, 0], dtype=np.float32)


@pytest.fixture
def color_khaki():
    return np.array([240, 230, 140], dtype=np.float32) / 255


@pytest.fixture(params=['color_white', 'color_yellow',
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
        [[0] * 3],
        [[0] * 3],
        [[1] * 3]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_white, *range_high)

    np.testing.assert_allclose(expected, result)


def test_channel_range_low(u16_3value_channel, color_white, range_low,
                           f32_3value_rgb_buffer):
    '''Extract low values from image in channel dictionary'''

    expected = np.array([
        [[0] * 3],
        [[1] * 3],
        [[1] * 3]
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_white, *range_low)

    np.testing.assert_allclose(expected, result)


def test_channel_color_red(u16_3value_channel, color_red, range_all,
                           f32_3value_rgb_buffer):
    '''Blend an image with one channel, testing red color'''

    expected = np.array([
        [[0] * 3],
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
        [[0] * 3],
        [[12345 / 65535] * 3],
        [[1] * 3]
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
        [[0] * 3],
        [[
            240 / 255 * (12345 / 65535),
            230 / 255 * (12345 / 65535),
            140 / 255 * (12345 / 65535)
        ]],
        [[
            240 / 255,
            230 / 255,
            140 / 255
        ]]
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
        [[0] * 3],
        [[
            240 / 255,
            230 / 255,
            140 / 255
        ]],
        [[
            240 / 255,
            230 / 255,
            140 / 255
        ]],
    ], dtype=np.float32)

    result = composite_channel(f32_3value_rgb_buffer, u16_3value_channel,
                               color_khaki, *range_low)

    np.testing.assert_allclose(expected, result)


def test_channels_two_channel(u16_checkered_channel,
                              u16_checkered_channel_inverse,
                              color_blue, color_yellow, range_all):
    '''Test blending an image with two channels'''

    expected = np.array([
        [[1, 1, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 1, 0]],
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
            'image': np.array([[0] * 3], dtype=np.uint16),
            'color': color_white,
            'min': range_all[0],
            'max': range_all[1]
        },
        {
            'image': np.array([[0] * 2], dtype=np.uint16),
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
