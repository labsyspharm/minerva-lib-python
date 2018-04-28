import numpy as np
import skimage.exposure


def handle_channel(channel):
    '''
    Arguments:
        channel: Dict to blend with rendering settings:
            {
                image: Numpy image data
                color: r, g, b float32 array within 0, 1
                min: Range minimum, float32 range 0, 1
                max: Range maximum, float32 range 0, 1
            }

    Returns:
        Threshholded float32 image normalzied within 0, 1
        r, g, b float32 array color within 0, 1
    '''
    f32_range = (channel['min'], channel['max'])
    f32_image = skimage.img_as_float(channel['image'])

    # Return rescaled image and color for additive blending
    f32_image = skimage.exposure.rescale_intensity(f32_image, f32_range)
    return f32_image, channel['color']


def linear_rgb(channels):
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
        if num_channels == 1:
            # Needs image to be 2D ndarray
            out_shape = image.shape + (3,)

            # Output buffer for blending
            out_buffer = np.zeros(out_shape, dtype=np.float32)

        # Additive blending for RGB buffer
        for idx, scalar in enumerate(color):

            # Add color for one channel in image buffer
            out_buffer[:, :, idx] += image * scalar

    # Must be at least one channel
    if num_channels < 1:
        raise ValueError('At least one channel must be specified')

    # Return gamma correct image within 0, 1
    np.clip(out_buffer, 0, 1, out=out_buffer)
    return out_buffer
