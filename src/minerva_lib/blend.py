import numpy as np
import skimage.exposure


def clip_image(channel):
    ''' Clip an image from min, max to 0,1

    Arguments:
        channel: Dict to blend with rendering settings:
            {
                image: Numpy 2D image data
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


def composite_channel(image, color):
    ''' Yield 3 image channels for r, g, b

    Arguments:
        image: Numpy 2D image data
        color: r, g, b float32 array within 0, 1

    Yields:
        Numpy 2D image data for r, g, b channels
    '''

    for scalar in color:
        yield image * scalar


def composite_channels(channels):
    '''Blend all channels into one normalized image

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

    # rescaled images and normalized colors
    for image, color in map(clip_image, channels):

        num_channels += 1
        if num_channels == 1:

            # Needs image to be 2D ndarray
            out_shape = image.shape + (3,)

            # Output buffer for blending
            out_buffer = np.zeros(out_shape, dtype=np.float32)

        # Add all three channels to output buffer
        rgb_image = composite_channel(image, color)
        out_buffer[:, :, 0] += next(rgb_image)
        out_buffer[:, :, 1] += next(rgb_image)
        out_buffer[:, :, 2] += next(rgb_image)

    # Must be at least one channel
    if num_channels < 1:
        raise ValueError('At least one channel must be specified')

    # Return gamma correct image within 0, 1
    np.clip(out_buffer, 0, 1, out=out_buffer)
    return skimage.exposure.adjust_gamma(out_buffer, 1 / 2.2)
