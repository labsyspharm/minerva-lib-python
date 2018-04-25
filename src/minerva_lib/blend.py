import numpy as np


def threshhold_image(image, min_, max_):
    ''' Use only pixel values above min_ below max_
    Arguments:
        image: ndarray modified in place
        min_: values here become zero
        max_: values here become max_ - min_
    '''
    # Set all values outside of range to zero
    np.clip(image, min_, max_, out=image)
    image -= min_


def scale_color(color, range_):
    ''' Color becomes conversion from inputs
    Arguments:
        color: b, g, r float array within 0, 1
        range_: extent of input range
    Returns:
        Color conversion from inputs to 0, 255
    '''
    # Embed conversion within colors
    return 255 * color / range_


def handle_channel(channel):
    """
    Arguments:
        channel: Dict to blend with rendering settings:
            {
                image: Numpy image data
                color: N-channel by b, g, r float32 color
                min: Range minimum, float32 range
                max: Range maximum, float32 range
            }

    Returns:
        Threshholded image values between min and max
        Scaled color to convert image to 24-bit BGR
    """
    image = channel['image']
    color = channel['color']
    input_limit = np.iinfo(image.dtype).max

    # Translate min and max to image integers
    min_ = int(round(channel['min'] * input_limit))
    max_ = int(round(channel['max'] * input_limit))

    # Prepare image and color for averaging
    threshhold_image(image, min_, max_)
    color_ = scale_color(color, max_ - min_)
    return (image, color_)


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

    # Final output and buffer for blending
    image_out = np.zeros(shape + (3,), dtype=np.uint8)
    image_buffer = np.zeros(shape, dtype=np.float32)

    # threshhold images and normalize colors
    images_colors = map(handle_channel, channels)
    images_colors = list(images_colors)
    total = len(images_colors)

    # colorize image
    for color_idx in range(3):
        for image, color in images_colors:

            # Add color for one channel in image buffer
            image_buffer += image * color[color_idx]

        np.round(image_buffer/total, out=image_buffer)

        # Write to output and reset buffer
        image_out[:, :, color_idx] = np.uint8(image_buffer)
        image_buffer *= 0

    return image_out
