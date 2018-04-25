import numpy as np


def handle_channel(channel):
    '''
    Arguments:
        channel: Dict to blend with rendering settings:
            {
                image: Numpy image data
                min: Range minimum, float32 range 0, 1
                max: Range maximum, float32 range 0, 1
            }

    Returns:
        Threshholded float32 image values within 0,1
        r, g, b float32 array color within 0, 1
    '''
    image = channel['image']
    input_limit = np.iinfo(image.dtype).max

    # Translate min and max to image integers
    min_ = channel['min'] * input_limit
    max_ = channel['max'] * input_limit

    # Return image for additive blending
    f32_image = np.float32(channel['image']) - min_
    np.clip(f32_image / (max_ - min_), 0, 1, out=f32_image)

    return f32_image, channel['color']


def linear_bgr(channels):
    '''Blend all channels given
    Arguments:
        channels: List of dicts of channels to blend with rendering settings:
            {
                image: Numpy image data
                color: r, g, b float32 array within 0, 1
                min: Range minimum, float32 range 0, 1
                max: Range maximum, float32 range 0, 1
            }

    Returns:
        float32 y by x by r, g, b color image within 0, 1
    '''
    num_channels = 0

    # threshhold images and normalize colors
    for image, color in map(handle_channel, channels):

        num_channels += 1
        if num_channels is 1:
            # Output buffer for blending
            out_shape = image.shape + (3,)
            out_buffer = np.zeros(out_shape, dtype=np.float32)

        # Additive blending for RGB buffer
        for idx, scalar in enumerate(color):

            # Add color for one channel in image buffer
            out_buffer[:, :, idx] += image * scalar

    # Must be at least one channel
    if num_channels < 1:
        raise ValueError('At least one channel must be specified')

    return out_buffer / num_channels
