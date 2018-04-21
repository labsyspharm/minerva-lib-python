import numpy as np


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
    for channel in channels:
        color = channel['color']
        image = channel['image']
        input_limit = np.iinfo(image.dtype).max

        # Translate min and max to image integers
        min_ = int(round(channel['min'] * input_limit))
        max_ = int(round(channel['max'] * input_limit))
        # Apply normalization to colors
        color *= 255 / len(channels)
        color /= (max_ - min_)

        # Set all values outside of range to zero
        image[(image < min_) | (image > max_)] = min_
        image -= min_

    # colorize image
    for color_idx in range(3):
        for channel in channels:
            color = channel['color']
            image = channel['image']

            # Add color for one channel in image buffer
            image_buffer += image * color[color_idx]

        # Write to output and reset buffer
        image_out[:, :, color_idx] = np.uint8(np.round(image_buffer))
        image_buffer *= 0

    return image_out
