import numpy as np
from .blend import composite_channels

def select_tiles(tile_size, origin, crop_size):
    ''' Select tile coordinates covering crop region

    Args:
        tile_size: width, height of one tile
        origin: x, y coordinates to begin selection
        crop_size: width, height to select

    Returns:
        List of integer i, j tile indices
    '''
    start = np.array(origin)
    end = start + crop_size
    fractional_start = start / tile_size
    fractional_end = end / tile_size

    # Round to get indices containing selection
    first_index = np.int64(np.floor(fractional_start))
    last_index = np.int64(np.ceil(fractional_end))

    # Calculate all indices between first and last
    index_shape = last_index - first_index
    offsets = np.argwhere(np.ones(index_shape))
    indices = first_index + offsets

    return indices.tolist()


def get_tile_bounds(indices, tile_size, origin, crop_size):
    ''' Define subregion to extract relative to tile

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        origin: x, y coordinates to begin selection
        crop_size: width, height to select

    Returns:
        start uv, end uv relative to tile
    '''

    crop_end = np.int64(origin) + crop_size
    tile_start = np.int64(indices) * tile_size
    tile_end = tile_start + tile_size

    # Relative to tile start
    return [
        np.maximum(origin, tile_start) - tile_start,
        np.minimum(tile_end, crop_end) - tile_start
    ]


def get_out_bounds(indices, tile_size, origin, crop_size):
    ''' Define position of cropped tile relative to origin

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        origin: x, y coordinates to begin selection
        crop_size: width, height to select

    Returns:
        start xy, end xy relative to origin
    '''

    crop_end = np.int64(origin) + crop_size
    tile_start = np.int64(indices) * tile_size
    tile_end = tile_start + tile_size

    # Relative to origin
    return [
        np.maximum(origin, tile_start),
        np.minimum(tile_end, crop_end)
    ]


def stitch_channels(out, tile_bounds, out_bounds, channels):
    ''' Position channels from tile into output image

    Args:
        out: 2D numpy array to contain stitched channels
        tile_bounds: start uv, end uv to get from tile
        out_bounds: start xy, end xy to put in _out_
        channels: List of dicts for channels to blend.
            Each dict in the list must have the
            following rendering settings:
            {
                image: Numpy 2D image data of any type
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }

    Returns:
        A reference to the modified _out_ array
    '''

    # Take data from tile
    composite = composite_channels(channels)
    [u0, v0], [u1, v1] = tile_bounds
    subregion = composite[v0:v1, u0:u1]

    # Draw data to output
    [x0, y0], [x1, y1] = out_bounds
    out[y0:y1, x0:x1] = subregion

    return out
