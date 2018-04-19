import numpy as np


def to_f32(img):
    '''Scale the dynamic range to 0.0 - 1.0

    Arguments:
    img: an integer image
    '''
    try:
        dtype_info = np.iinfo(img.dtype)
        assert dtype_info.min is 0
    except (ValueError, AssertionError):
        raise ValueError('Scaling to 0,1 requires unsigned integers')

    one = dtype_info.max + 1
    return np.float32(img / one)


def f32_to_bgr(f_img, color=[1, 1, 1]):
    '''Reshape into a color image

    Arguments:
    f_img: float32 image to reshape
    '''

    try:
        assert np.all((f_img >= 0) & (f_img <= 1))
    except AssertionError:
        raise ValueError('Color image requires values from 0,1')

    # Give the image a color dimension
    f_vol = f_img[:, :, np.newaxis]
    f_bgr = np.repeat(f_vol, 3, 2) * color
    return (256 * f_bgr).astype(np.uint8)


def linear_bgr(all_imgs, colors, ranges):
    '''Blend all channels given
    Arguments:
        all_imgs: List of numpy images, one per channel
        colors: N-channel by b, g, r float32 color
        ranges: N-channel by min, max float32 range

    Returns:
        uint8 y by x by 3 color BGR image
    '''

    num_channels = len(all_imgs)

    # Ensure a color per channel has been specified
    if num_channels != len(colors):
        raise ValueError('One color per channel must be specified')

    # Ensure a range per channel has been specified
    if num_channels != len(ranges):
        raise ValueError('One range per channel must be specified')

    # Get the shape of the pixels and ensure they are all the same size
    shape = all_imgs[0].shape
    for pixels in all_imgs:
        if pixels.shape != shape:
            raise ValueError('All channels must have equal pixel dimensions')

    # Shape of 3 color pixels
    shape_color = shape + (3,)

    # Final buffer for blending
    img_buffer = np.zeros(shape_color, dtype=np.float32)

    # Process all channels
    for color, range, pixels in zip(colors, ranges, all_imgs):

        # Scale the dynamic range
        img_ranged = to_f32(pixels)

        # Maximum color for this channel
        avg_factor = 1.0 / num_channels
        color_factor = color * avg_factor

        # Fraction of full range
        lowest, highest = range
        clip_size = highest - lowest

        # Apply the range
        img_ranged[(img_ranged < lowest) | (img_ranged > highest)] = lowest
        img_norm = (img_ranged - lowest) / clip_size

        # Add the colored data to the image
        y_shape, x_shape = img_norm.shape
        img_color = f32_to_bgr(img_norm, color_factor)
        img_buffer[0:y_shape, 0:x_shape] += img_color

    return np.uint8(img_buffer)
