from functools import reduce
import numpy as np
from .blend import composite_channel
from . import skimage_inline as ski


def scale_image_nearest_neighbor(source, factor):
    ''' Resizes an image by a given factor using nearest neighbor pixels.

    Arguments:
        source: A 2D or 3D numpy array to resize
        factor: The ratio of `out` shape over `source` shape

    Returns:
        A numpy array with the resized `source`
    '''

    if factor <= 0:
        raise ValueError('Scale factor must be above zero')

    factors = [factor, factor, 1]

    out_shape = [int(round(a * b)) for a, b in zip(source.shape, factors)]
    out_max_x = out_shape[1] - 1
    out_max_y = out_shape[0] - 1
    out = np.zeros(out_shape)

    for source_x in range(0, source.shape[1]):
        for source_y in range(0, source.shape[0]):
            out_x = min(int(round(source_x * factor)), out_max_x)
            out_y = min(int(round(source_y * factor)), out_max_y)
            out[out_y, out_x] = source[source_y, source_x]

    return out


def get_optimum_pyramid_level(input_shape, level_count,
                              max_size=None, round_up=True):
    ''' Gives a pyramid level that scales a region near a maximum size.

    Arguments:
        input_shape: The width, height at pyramid level 0
        level_count: Number of available pyramid levels
        max_size: Maximum output image extent in x or y
        round_up: Use higher-magnifcation pyramid level contained by
                  `max_size` if true. Use lower-magnification pyramid
                  level containing `max_size` if false.

    Returns:
        Integer power of 2 pyramid level
    '''

    if max_size is None:
        return 0

    longest_side = max(*input_shape)
    ratio_log = np.log2(longest_side / max_size)

    if round_up:
        level = np.ceil(ratio_log)
    else:
        level = np.floor(ratio_log)

    return int(np.clip(level, 0, level_count - 1))


def scale_by_pyramid_level(coordinates, level):
    ''' Returns the coordinates measured at a given pyramid level.

    Arguments:
        coordrnates: Coordinates to downscale by _level_
        level: Integer power of 2 pyramid level

    Returns:
        downscaled integer coordinates
    '''

    scaled_coords = np.array(coordinates) / (2 ** level)
    return np.int64(np.floor(scaled_coords))


def get_tile_start(tile_size, crop_origin):
    ''' Returns the indices of the first tile in the region.

    Args:
        tile_size: width, height of one tile
        crop_origin: x, y origin of requested image region
        crop_size: width, height of requested image region

    Returns:
        Two-item int64 array i, j tile indices
    '''
    start = np.array(crop_origin)

    return np.int64(np.floor(start / tile_size))


def get_tile_count(tile_size, crop_origin, crop_size):
    ''' Returns the number of tiles in the region.

    Args:
        tile_size: width, height of one tile
        crop_origin: x, y origin of requested image region
        crop_size: width, height of requested image region
        image_size: width, height of full image

    Returns:
        Two-item int64 array i, j tile indices
    '''
    end = np.array(crop_origin) + crop_size

    return np.int64(np.ceil(end / tile_size))


def get_subregion(indices, tile_size, crop_origin, crop_size):
    '''Determines the region within this specific tile
    that is required for the requested image region.

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        crop_origin: x, y origin of requested image region
        crop_size: width, height of requested image region

    Returns:
        The part of the tile within the requested region
        Subregion start, end relative to origin of tile
    '''

    tile_start = np.int64(indices) * tile_size
    crop_end = np.int64(crop_origin) + crop_size
    tile_end = tile_start + tile_size

    # Should be true that 0 <= start_uv < end_uv <= tile_size
    start_uv = np.maximum(crop_origin, tile_start) - tile_start
    end_uv = np.minimum(tile_end, crop_end) - tile_start

    return [start_uv, end_uv]


def get_position(indices, tile_size, crop_origin):
    ''' Determines where in the requested image to insert
    the required region from this specific tile.

    Args:
        indices: integer i, j tile indices
        tile_size: width, height of one tile
        crop_origin: x, y origin of requested image region

    Returns:
        Position of tile region within requested image region
    '''

    tile_start = np.int64(indices) * tile_size

    # Should never be negative
    return np.maximum(crop_origin, tile_start) - crop_origin


def validate_region_bounds(crop_origin, crop_size, image_size):
    ''' Decides if the image contains the requested image region.

    Args:
        crop_origin: x, y origin of requested image region
        crop_size: width, height of requested image region
        image_size: width, height of full image

    Returns:
        True if the requested image region is valid
    '''

    if any(np.less_equal(crop_size, 0)):
        return False

    if any(np.less(crop_origin, 0)):
        return False

    crop_end = np.array(crop_origin) + crop_size
    if any(np.less(image_size, crop_end)):
        return False

    return True


def select_tiles(tile_size, crop_origin, crop_size):
    ''' Selects tile indices to fill the requested region.

    Args:
        tile_size: width, height of one tile
        crop_origin: x, y origin of requested image region
        crop_size: width, height of requested image region

    Returns:
        List of integer i, j tile indices
    '''
    start_ij = get_tile_start(tile_size, crop_origin)
    count_ij = get_tile_count(tile_size, crop_origin, crop_size)

    # Calculate all indices between first and last
    offsets = np.argwhere(np.ones(count_ij - start_ij))
    return (start_ij + offsets).tolist()


def stitch_tile(out, subregion, position, tile):
    ''' Positions this tile into the requested region.

    Args:
        out: 2D numpy array to contain stitched channels
        subregion: The part of the tile within the requested region
        position: Origin of tile in requested region
        tile: 2D numpy array to stitch within _out_

    Returns:
        A reference to _out_
    '''

    # Take subregion from tile
    [u_0, v_0], [u_1, v_1] = subregion
    subtile = tile[v_0:v_1, u_0:u_1]
    shape = np.int64(subtile.shape)

    # Define boundary
    x_0, y_0 = position
    y_1, x_1 = [y_0, x_0] + shape[:2]

    # Assign subregion within boundary
    out[y_0:y_1, x_0:x_1] += subtile

    return out


def stitch_tiles(tiles, tile_size, crop_origin, crop_size):
    ''' Positions all image tiles and channels in the requested region.

    Args:
        tiles: Iterator of tiles to blend. Each dict in the
            list must have the following rendering settings:
            {
                indices: Integer i, j tile indices
                image: Numpy 2D image data of any type
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        tile_size: width, height of one tile
        crop_origin: x, y origin of requested image region
        crop_size: width, height of requested image region

    Returns:
        For a given `shape` of `(width, height)`,
        returns a float32 RGB color image with shape
        `(height, width, 3)` and values in the range 0 to 1
    '''
    inputs = [t for t in tiles if t]
    rgb_tile_size = tuple(tile_size) + (3,)
    stitched = np.zeros(tuple(crop_size) + (3,))

    def composite(tile_hash, tile):
        ''' Composite all tiles with same indices
        '''

        idx = tuple(tile['indices'])

        if idx not in tile_hash:
            tile_buffer = {
                'indices': idx,
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

        image = tile['image']
        idx = tuple(tile['indices'])

        subregion = get_subregion(idx, tile_size, crop_origin, crop_size)
        position = get_position(idx, tile_size, crop_origin)

        return stitch_tile(array, subregion, position, image)

    # Composite all tiles with same indices and stitch
    composited = reduce(composite, inputs, {}).values()
    stitched = reduce(stitch, composited, stitched)

    # Return gamma correct image within 0, 1
    np.clip(stitched, 0, 1, out=stitched)
    return ski.adjust_gamma(stitched, 1 / 2.2)
