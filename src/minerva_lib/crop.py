import numpy as np
import skimage.exposure
from .blend import composite_channel
from functools import reduce


def get_lod(lods, max_size, width, height):
    ''' Calculate the level of detail

    Arguments:
        lods: Number of available levels of detail
        max_size: Maximum image extent in x or y
        width: Extent of image in x
        height: Extent of image in y

    Returns:
        Integer power of 2 level of detail
    '''

    longest_side = max(width, height)
    lod = np.ceil(np.log2(longest_side / max_size))
    return int(np.clip(lod, 0, lods - 1))


def apply_lod(coordinates, lod):
    ''' Apply the level of detail to coordinates

    Arguments:
        coordinates: Coordinates to downscale by _lod_
        lod: Integer power of 2 level of detail

    Returns:
        downscaled integer coordinates
    '''

    scaled_coords = np.array(coordinates) / (2 ** lod)
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


def stitch_tiles(tiles, tile_size, crop_size, order='before'):
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
        order: Composite `'before'` or `'after'` stitching

    Returns:
        For a given `shape` of `(width, height)`,
        returns a float32 RGB color image with shape
        `(height, width, 3)` and values in the range 0 to 1
    '''
    def stitch(a, t):
        return stitch_tile(a, t['subregion'], t['position'], t['image'])

    def composite(a, t):
        return composite_channel(a, t['image'], t['color'],
                                 t['min'], t['max'], a)

    class Group():

        composite_keys = {'color', 'min', 'max'}
        stitch_keys = {'position', 'subregion'}

        if order == 'before':
            size = tuple(tile_size) + (3,)
            dtype = staticmethod(lambda t: np.float32)
            index = staticmethod(lambda t: tuple(t['indices']))
            first_call = staticmethod(composite)
            second_call = staticmethod(stitch)
            in_keys = composite_keys
            out_keys = stitch_keys

        if order == 'after':
            size = tuple(crop_size)
            dtype = staticmethod(lambda t: t['image'].dtype)
            index = staticmethod(lambda t: t['channel'])
            first_call = staticmethod(stitch)
            second_call = staticmethod(composite)
            in_keys = stitch_keys
            out_keys = composite_keys

        def __init__(self, t):
            d = self.dtype(t)
            self.buffer = {k: t[k] for k in self.out_keys}
            self.buffer['image'] = np.zeros(self.size, dtype=d)
            self.inputs = []
            self += t

        def __iadd__(self, t):
            self.inputs += [
                {k: t[k] for k in self.in_keys | {'image'}}
            ]
            return self

    def hash_groups(groups, tile):
        '''
        If before: group channels by tile
        If after: group tiles by channel
        '''

        idx = Group.index(tile)

        if idx not in groups:
            groups[idx] = Group(tile)
        else:
            groups[idx] += tile

        return groups

    def combine_groups(out, group):
        '''
        If before: Composite to RGBA float tile then stitch
        If after: Stitch to gray integer image then composite
        '''
        for t in group.inputs:
            group.first_call(group.buffer['image'], t)
        group.second_call(out, group.buffer)

        return out

    inputs = [t for t in tiles if t]
    out = np.zeros(tuple(crop_size) + (3,))

    # Make groups by channel or by tile
    groups = reduce(hash_groups, inputs, {}).values()
    # Stitch and Composite in either order
    out = reduce(combine_groups, groups, out)

    # Return gamma correct image within 0, 1
    np.clip(out, 0, 1, out=out)
    return skimage.exposure.adjust_gamma(out, 1 / 2.2)


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
    for indices in select_tiles(tile_size, origin, crop_size):

        position = get_position(indices, tile_size, origin)
        subregion = get_subregion(indices, tile_size,
                                  origin, crop_size)

        for cid, channel in enumerate(channels):

            yield {
                'channel': cid,
                'indices': indices,
                'position': position,
                'subregion': subregion,
                'min': channel['min'],
                'max': channel['max'],
                'color': list(channel['color']),
            }
