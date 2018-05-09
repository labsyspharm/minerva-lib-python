import numpy as np
import skimage.exposure


def composite_channel(a, image, color, range_min, range_max, out=None):
    ''' Composite given image _a_ with rendered channel _image_

    To update _a_ destructively, pass the same array to _a_ and _out_.

    Args:
        a: Numpy array to composite
        image: Numpy array of channel to render and composite
        color: Color as r, g, b float32 array within 0, 1
        range_min: Threshhold range minimum, float32 within 0, 1
        range_max: Threshhold range maximum, float32 within 0, 1
        out: Optional output numpy array in which to place the result.

    Returns:
        A numpy array with the same shape as the composited image.
        If an output array is specified, a reference to _out_ is returned.
    '''

    if out is None:
        out = a.copy()

    # Rescale the new channel to a float64 between 0 and 1
    f64_range = (range_min, range_max)
    f64_image = skimage.img_as_float(image)
    f64_image = skimage.exposure.rescale_intensity(f64_image, f64_range)

    # Colorize and add the new channel to composite image
    for i, component in enumerate(color):
        out[:, :, i] += f64_image * component

    return out


def composite_channels(channels):
    '''Blend all channels into one normalized image

    Arguments:
        channels: List of dicts of channels to blend with rendering settings:
            {
                image: Numpy 2D image data of any type
                color: Color as r, g, b float32 array within 0, 1
                min: Threshhold range minimum, float32 within 0, 1
                max: Threshhold range maximum, float32 within 0, 1
            }

    Returns:
        An r, g, b float32 color image. Each color component has the same \
shape as each image in channels with color component values within 0, 1.
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
        args = map(channel.get, ['image', 'color', 'min', 'max'])
        out_buffer = composite_channel(out_buffer, *args, out=out_buffer)

    # Return gamma correct image within 0, 1
    np.clip(out_buffer, 0, 1, out=out_buffer)
    return skimage.exposure.adjust_gamma(out_buffer, 1 / 2.2)
