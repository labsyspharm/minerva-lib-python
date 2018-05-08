import numpy as np
import skimage.exposure


def composite_channel(in_array, channel, out_array=None):
    ''' Yield 3 image channels for r, g, b

    Arguments:
        in_array: Previously composited 2D image
        channel: Channel to add to newly composited 2D image
            {
                image: Numpy 2D image data
                color: r, g, b float32 array within 0, 1
                min: Range minimum, float32 range 0, 1
                max: Range maximum, float32 range 0, 1
            }
        out_array: Location of newly composited 2D image

    Returns:
        Newly composited 2D image stored in out_array if given
    '''
    if out_array is None:
        out_array = in_array.copy()

    # Rescale the new channel to a float64 between 0 and 1
    f64_range = (channel['min'], channel['max'])
    f64_image = skimage.img_as_float(channel['image'])
    f64_image = skimage.exposure.rescale_intensity(f64_image, f64_range)

    # Colorize and add the new channel to composite image
    for i, component in enumerate(channel['color']):
        out_array[:, :, i] += f64_image * component

    return out_array


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
    out_buffer = np.zeros(shape_color, dtype=np.float32)

    # rescaled images and normalized colors
    for channel in channels:

        # Add all three channels to output buffer
        out_buffer = composite_channel(out_buffer, channel, out_buffer)

    # Return gamma correct image within 0, 1
    np.clip(out_buffer, 0, 1, out=out_buffer)
    return skimage.exposure.adjust_gamma(out_buffer, 1 / 2.2)
