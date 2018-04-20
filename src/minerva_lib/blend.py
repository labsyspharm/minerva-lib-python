import numpy as np


def to_f32(img):
    '''Scale the dynamic range to 0.0 - 1.0

    Arguments:
    img: An integer image
    '''

    # No well-defined behavior for decimal values or values less than 0
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

    # All inputs should be normalized between 0 and 1
    try:
        assert np.all((f_img >= 0) & (f_img <= 1))
    except AssertionError:
        raise ValueError('Color image requires values from 0,1')

    # Give the image a color dimension
    f_vol = f_img[:, :, np.newaxis]
    f_bgr = np.repeat(f_vol, 3, 2) * color
    return (256 * f_bgr).astype(np.uint8)


def linear_bgr(channels):
    '''Blend all channels given
    Arguments:
        channels: List of dicts of channels to blend with rendering settings:
            {
                image: Numpy image data
                color: N-channel by b, g, r float32 color
                min: Range minimum, float32 range
                max: Range maximum, float32 range
            }

    Returns:
        uint8 y by x by 3 color BGR image
    '''

    # Get the number of channels
    num_channels = len(channels)

    # Must be at least one channel
    if num_channels < 1:
        raise ValueError('At least one channel must be specified')

    # Ensure that dimensions of all channels are equal
    shape = channels[0]['image'].shape
    for channel in channels:
        if channel['image'].shape != shape:
            raise ValueError('All channel images must have equal dimensions')

    # Shape of 3 color image
    shape_color = shape + (3,)

    # Final buffer for blending
    image_buffer = np.zeros(shape_color, dtype=np.float32)

    # Process all channels
    for channel in channels:

        image = channel['image']
        color = channel['color']
        min_ = channel['min']
        max_ = channel['max']

        # Scale the dynamic range
        img_ranged = to_f32(image)

        # Maximum color for this channel
        avg_factor = 1.0 / num_channels
        color_factor = color * avg_factor

        # Fraction of full range
        clip_size = max_ - min_

        # Apply the range
        img_ranged[(img_ranged < min_) | (img_ranged > max_)] = min_
        img_norm = (img_ranged - min_) / clip_size

        # Add the colored data to the image
        y_shape, x_shape = img_norm.shape
        img_color = f32_to_bgr(img_norm, color_factor)
        image_buffer[0:y_shape, 0:x_shape] += img_color

    return np.uint8(image_buffer)
