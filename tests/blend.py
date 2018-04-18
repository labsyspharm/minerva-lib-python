""" Compare blend results with expected output
"""
import numpy as np
from minerva_lib.blend import to_f32
from minerva_lib.blend import linear_bgr


class Key(object):
    """ Constants used for testing
    """
    # Sample named ranges
    range_all = np.float32([0, 1])
    range_hi = np.float32([0.5, 1])
    range_lo = np.float32([0, 256./65535.])

    # Sample named colors
    white = np.float32([1, 1, 1])
    yellow = np.float32([0, 1, 1])
    green = np.float32([0, 1, 0])
    blue = np.float32([1, 0, 0])
    red = np.float32([0, 0, 1])

    @staticmethod
    def sanity_check(t_pair):
        """ Compare basic details of two images

        Arguments:
            t_pair: list of two images to compare
        """
        type_pair = [x.dtype for x in t_pair]
        shape_pair = [x.shape for x in t_pair]

        # Assume data type and shape
        assert len(set(type_pair)) == 1
        assert not np.subtract(*shape_pair).any()

    @staticmethod
    def full_check(t_pair):
        """ Expect two images to be idential

        Arguments:
            t_pair: two arrays assumed identical
        """
        # Assume pixel-for-pixel image
        assert not np.subtract(*t_pair).any()


def generic_test_tile(t_chans, t_ok, t_keys, t_list=None):
    """ Run test on tile blend function

    Arguments:
        t_keys: keywords for call
        t_chans: list of input channels
        t_ok: assumed output image
        t_list: list of tests to run
    """
    # Blend all input tiles
    t_out = linear_bgr(t_chans, **t_keys)
    t_pair = t_ok, t_out

    # Run standard tests by default
    if not t_list:
        t_list = [
            Key.sanity_check,
            Key.full_check,
        ]
    for t_fn in t_list:
        t_fn(t_pair)


def easy_test_tile(t_r, t_c, t_in, t_ok, t_list=None):
    """ Combine one channel to expected output and compare

    Arguments:
        t_r: 2 min,max float32
        t_c: 3 b,g,r float32
        t_in: input channel
        t_ok: expected output image
        t_list: list of tests to run
    """
    t_keys = {
        'ranges': t_r[np.newaxis],
        'colors': t_c[np.newaxis]
    }
    generic_test_tile([t_in], t_ok, t_keys, t_list)


def many_test_tile(ranges, colors, t_chans, t_ok, t_list=None):
    """ Combine many channels to expected output and compare

    Arguments:
        ranges: N channels by 2 min,max float32
        colors: N channels by 3 b,g,r float32
        t_ok: expected output image
        t_chans: list of input channels
        t_list: list of tests to run
    """
    t_keys = {
        'ranges': ranges,
        'colors': colors
    }
    generic_test_tile(t_chans, t_ok, t_keys,  t_list)


# _________________________
# Actual pytest entrypoints

def test_tile_1channel_gray():
    """ 1 channel cut and color
    """
    # Sample range of u16
    t_in = np.uint16([
        [0],
        [256],
        [65535],
    ])

    # START TEST
    t_ok = np.uint8([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[255, 255, 255]],
    ])
    # Check mapping all values to white
    easy_test_tile(Key.range_all, Key.white, t_in, t_ok)

    # START TEST
    t_ok = np.uint8([
        [[0, 0, 0]],
        [[255, 255, 255]],
        [[0, 0, 0]],
    ])
    # Check mapping low values to white
    easy_test_tile(Key.range_lo, Key.white, t_in, t_ok)

    # START TEST
    t_ok = np.uint8([
        [[0, 0, 0]],
        [[0, 0, 0]],
        [[0, 255, 0]],
    ])
    # Check mapping high values to green
    easy_test_tile(Key.range_hi, Key.green, t_in, t_ok)


def test_tile_2channel_chess():
    """ 2 channel cut and color
    """
    ranges_all = np.stack((Key.range_all,)*2)
    blu_yel = np.stack((Key.blue, Key.yellow))
    # On/off grid
    t_chans = np.uint16([[
        [0, 65535],
        [65535, 0],
    ], [
        [65535, 0],
        [0, 65535],
    ]])
    # START TEST
    t_ok = 127*np.uint8([
        [Key.yellow, Key.blue],
        [Key.blue, Key.yellow],
    ])
    # Make sure blue/yellow grid has no overlaps
    many_test_tile(ranges_all, blu_yel, t_chans, t_ok)


def test_to_f32_full():
    ''' Test f32 conversion across full range
    '''
    # Sample range from 0 to 65535
    img_in = np.uint16([
        [0],
        [256],
        [65535],
    ])

    # The same range from 0 to 1
    img_ok = np.float32([
        [0.0],
        [256.0 / 65536.0],
        [65535.0 / 65536.0],
    ])

    # Check to_f32 handles uint16 range
    img_out = to_f32(img_in)
    img_pair = img_ok, img_out
    Key.full_check(img_pair)
