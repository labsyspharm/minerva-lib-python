import numpy as np
import skimage.exposure
from .blend import composite_channel
from functools import reduce


def get_optimum_pyramid_level(input_shape, level_count, max_size):
    ''' Calculate the pyramid level below a maximum

    Arguments:
        level_count: Number of available pyramid levels
        max_size: Maximum output image extent in x or y
        input_shape: The width, height at pyramid level 0

    Returns:
        Integer power of 2 pyramid level
    '''

    longest_side = max(*input_shape)
    level = np.ceil(np.log2(longest_side / max_size))
    return int(np.clip(level, 0, level_count - 1))


def scale_by_pyramid_level(coordinates, level):
    ''' Apply the pyramid level to coordinates

    Arguments:
        coordinates: Coordinates to downscale by _level_
        level: Integer power of 2 pyramid level

    Returns:
        downscaled integer coordinates
    '''

    scaled_coords = np.array(coordinates) / (2 ** level)
    return np.int64(np.floor(scaled_coords))


def select_tiles(tile_size, origin, crop_size):
    ''' Select tile coordinates covering crop region

    Args:
        tile_size: width, height of one tile
        origin: x, y coordinates to begin subregion
        crop_size: width, height to select

    Returns:
        List of integer i, j tile indices
    '''
    start = np.array(origin)
    end = start + crop_size
    fractional_start = start / tile_size
    fractional_end = end / tile_size

    # Round to get indices containing subregion
    first_index = np.int64(np.floor(fractional_start))
    last_index = np.int64(np.ceil(fractional_end))

    # Calculate all indices between first and last
    index_shape = last_index - first_index
    offsets = np.argwhere(np.ones(index_shape))
    indices = first_index + offsets

    return indices.tolist()


def get_subregion(indices, tile_size, origin, crop_size):
    ''' Define subregion to select from within tile

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        origin: x, y coordinates to begin subregion
        crop_size: width, height to select

    Returns:
        start uv, end uv relative to tile
    '''

    crop_end = np.int64(origin) + crop_size
    tile_start = np.int64(indices) * tile_size
    tile_end = tile_start + tile_size

    return [
        np.maximum(origin, tile_start) - tile_start,
        np.minimum(tile_end, crop_end) - tile_start
    ]


def get_position(indices, tile_size, origin):
    ''' Define position of cropped tile relative to origin

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        origin: x, y coordinates to begin subregion

    Returns:
        The xy position relative to origin
    '''

    tile_start = np.int64(indices) * tile_size

    return np.maximum(origin, tile_start) - origin


def stitch_tile(out, subregion, position, tile):
    ''' Position image tile into output array

    Args:
        out: 2D RGB numpy array to contain stitched channels
        subregion: Start uv, end uv to get from tile
        position: Origin of tile when composited in _out_
        tile: 2D numpy array to stitch within _out_

    Returns:
        A reference to _out_
    '''

    # Take subregion from tile
    [u0, v0], [u1, v1] = subregion
    subtile = tile[v0:v1, u0:u1]
    shape = np.int64(subtile.shape)

    # Define boundary
    x0, y0 = position
    y1, x1 = [y0, x0] + shape[:2]

    # Assign subregion within boundary
    out[y0:y1, x0:x1] += subtile

    return out


def stitch_tiles(tiles, tile_size, crop_size):
    ''' Position all image tiles for all channels

    Args:
        tiles: Iterator of tiles to blend. Each dict in the
            list must have the following rendering settings:
            {
                channel: Integer channel index
                indices: Integer i, j tile indices
                image: Numpy 2D image data of any type
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
                subregion: The start uv, end uv relative to tile
                position: The xy position relative to origin
            }
        tile_size: width, height of one tile
        crop_size: The width, height of output image

    Returns:
        For a given `shape` of `(width, height)`,
        returns a float32 RGB color image with shape
        `(height, width, 3)` and values in the range 0 to 1
    '''
    inputs = [t for t in tiles if t]
    stitched = np.zeros(tuple(crop_size) + (3,))
    rgb_tile_size = tuple(tile_size) + (3,)

    def composite(tile_hash, tile):
        ''' Composite all tiles with same indices
        '''

        idx = tuple(tile['indices'])

        if idx not in tile_hash:
            tile_buffer = {
                'position': tile['position'],
                'subregion': tile['subregion'],
                'image': np.zeros(rgb_tile_size, dtype=np.float32)
            }
            tile_hash[idx] = tile_buffer

        tile_h, tile_w = tile['image'].shape

        composite_channel(tile_hash[idx]['image'][:tile_h, :tile_w],
                          tile['image'], tile['color'],
                          tile['min'], tile['max'],
                          tile_hash[idx]['image'][:tile_h, :tile_w])

        return tile_hash

    def stitch(array, tile):
        ''' Stitch tile into an array
        '''
        return stitch_tile(array, tile['subregion'],
                           tile['position'], tile['image'])

    # Composite all tiles with same indices and stitch
    composited = reduce(composite, inputs, {}).values()
    stitched = reduce(stitch, composited, stitched)

    # Return gamma correct image within 0, 1
    np.clip(stitched, 0, 1, out=stitched)
    return skimage.exposure.adjust_gamma(stitched, 1 / 2.2)


def stitch_tiles_at_level(channels, tile_size, full_size, level):
    ''' Position all image tiles for all channels

    Args:
        tiles: Iterator of tiles to blend. Each dict in the
            list must have the following rendering settings:
            {
                channel: Integer channel index
                indices: Integer i, j tile indices
                image: Numpy 2D image data of any type
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
                subregion: The start uv, end uv relative to tile
                position: The xy position relative to origin
            }
        tile_size: width, height of one tile
        full_size: full-resolution width, height to select
        level: integer power of 2 pyramid level

    Returns:
        For a given `shape` of `(width, height)`,
        returns a float32 RGB color image with shape
        `(height, width, 3)` and values in the range 0 to 1
    '''

    crop_size = scale_by_pyramid_level(full_size, level)

    return stitch_tiles(channels, tile_size, crop_size)


def iterate_tiles(channels, tile_size, origin, crop_size):
    ''' Return crop settings for channel tiles

    Args:
        channels: An iterator of dicts for channels to blend. Each
            dict in the list must have the following settings:
            {
                channel: Integer channel index
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        tile_size: width, height of one tile
        origin: x, y coordinates to begin subregion
        crop_size: width, height to select

    Returns:
        An iterator of tiles to render for the given region.
        Each dict in the list has the following settings:
        {
            channel: Integer channel index
            indices: Integer i, j tile indices
            color: Color as r, g, b float array within 0, 1
            min: Threshhold range minimum, float within 0, 1
            max: Threshhold range maximum, float within 0, 1
        }
    '''

    for channel in channels:

        (r, g, b) = channel['color']
        _id = channel['channel']
        _min = channel['min']
        _max = channel['max']

        for indices in select_tiles(tile_size, origin, crop_size):

            (i, j) = indices
            (x0, y0) = get_position(indices, tile_size, origin)
            (u0, v0), (u1, v1) = get_subregion(indices, tile_size,
                                               origin, crop_size)

            yield {
                'channel': _id,
                'indices': (i, j),
                'position': (x0, y0),
                'subregion': ((u0, v0), (u1, v1)),
                'color': (r, g, b),
                'min': _min,
                'max': _max,
            }


def list_tiles_at_level(channels, tile_size,
                        full_origin, full_size, level):
    ''' Return crop settings all tiles at given level

    Args:
        channels: An iterator of dicts for channels to blend. Each
            dict in the list must have the following settings:
            {
                channel: Integer channel index
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        tile_size: width, height of one tile
        full_origin: full-resolution x, y coordinates to begin subregion
        full_size: full-resolution width, height to select
        level: integer power of 2 pyramid level

    Returns:
        An iterator of tiles to render for the given region.
        Each dict in the list has the following settings:
        {
            level: given pyramid level
            channel: Integer channel index
            indices: Integer i, j tile indices
            color: Color as r, g, b float array within 0, 1
            min: Threshhold range minimum, float within 0, 1
            max: Threshhold range maximum, float within 0, 1
        }
    '''

    origin = scale_by_pyramid_level(full_origin, level)
    crop_size = scale_by_pyramid_level(full_size, level)

    tiles = iterate_tiles(channels, tile_size, origin, crop_size)

    return [{**t, 'level': level} for t in tiles]
