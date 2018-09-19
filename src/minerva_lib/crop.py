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

    # The output will have the same number of color channels as the source
    out_shape = [int(round(a * b)) for a, b in zip(source.shape, factors)]
    out_max_x = out_shape[1] - 1
    out_max_y = out_shape[0] - 1
    output = np.zeros(out_shape)

    for source_x in range(0, source.shape[1]):
        for source_y in range(0, source.shape[0]):
            out_x = min(int(round(source_x * factor)), out_max_x)
            out_y = min(int(round(source_y * factor)), out_max_y)
            output[out_y, out_x] = source[source_y, source_x]

    return output


def get_optimum_pyramid_level(input_shape, level_count,
                              output_size, prefer_higher_resolution):
    ''' Returns nearest available pyramid level for a resolution near the
        ideal resolution needed to render an full resolution image region
        at the desired downsampled output size.

    Arguments:
        input_shape: Tuple of integer height, width at full resolution
        level_count: Integer number of available pyramid levels
        output_size: Integer length of output image longest dimension
        prefer_higher_resolution: Set True to return the pyramid level
            for a resolution exceeding or equal to the ideal resolution.
            Set False to return the pyramid level for a resolution
            less than or equal to the ideal resolution.

    Returns:
        Integer power of 2 pyramid level
    '''

    longest_side = max(*input_shape)
    ratio_log = np.log2(longest_side / output_size)

    if prefer_higher_resolution:
        level = np.floor(ratio_log)
    else:
        level = np.ceil(ratio_log)

    return int(np.clip(level, 0, level_count - 1))


def scale_by_pyramid_level(coordinates, level):
    ''' Divides the given coordinates to match the given power of 2 level.

    Arguments:
        coordinates: Tuple of integer coordinates to downscale
        level: Integer power of 2 pyramid level

    Returns:
        Tuple of downscaled integer coordinates
    '''

    scaled_coords = np.array(coordinates) / (2 ** level)
    integer_coords = np.int64(np.round(scaled_coords))
    return tuple(integer_coords.tolist())


def get_tile_start(tile_shape, output_origin):
    ''' Returns the indices of the first tile in the region.

    Args:
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image

    Returns:
        Two-item int64 array of y, x tile indices
    '''
    start = np.array(output_origin)

    return np.int64(np.floor(start / tile_shape))


def get_tile_count(tile_shape, output_origin, output_shape):
    ''' Count number of tiles along height, width of a region.

    Args:
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image
        output_shape: Tuple of integer height, width of resulting image

    Returns:
        Two-item int64 array tile count along height, width
    '''
    end = np.array(output_origin) + output_shape

    return np.int64(np.ceil(end / tile_shape))


def select_subregion(indices, tile_shape, output_origin, output_shape):
    '''Determines the region within this specific tile
    that is required for the resulting image.

    Args:
        indices: Tuple of integer y, x tile indices
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image
        output_shape: Tuple of integer height, width of resulting image

    Returns:
        Start y, x and end y, x integer coordinates for the
            part of the tile needed for the resulting image
    '''

    tile_start = np.int64(indices) * tile_shape
    output_end = np.int64(output_origin) + output_shape
    tile_end = tile_start + tile_shape

    # at the start of the tile or the start of the requested region
    y_0, x_0 = np.maximum(output_origin, tile_start) - tile_start
    # at the end of the tile or the end of the requested region
    y_1, x_1 = np.minimum(tile_end, output_end) - tile_start

    return (y_0, x_0), (y_1, x_1)


def select_position(indices, tile_shape, output_origin):
    ''' Determines where in the resulting image to insert
    the required region from this specific tile.

    Args:
        indices: Tuple of integer y, x tile indices
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image

    Returns:
        Tuple of integer y, x position of tile region within resulting image
    '''

    tile_start = np.int64(indices) * tile_shape

    # At the start of the tile or the start of the requested region
    y_0, x_0 = np.maximum(output_origin, tile_start) - output_origin

    return (y_0, x_0)


def validate_region_bounds(output_origin, output_shape, image_shape):
    ''' Determines if the image contains the resulting image.

    Args:
        output_origin: Tuple of integer y, x origin of resulting image
        output_shape: Tuple of integer height, width of resulting image
        image_shape: Tuple of integer height, width of full image

    Returns:
        True if the resulting image is valid
    '''

    if any(np.less_equal(output_shape, 0)):
        return False

    if any(np.less(output_origin, 0)):
        return False

    output_end = np.array(output_origin) + output_shape
    if any(np.less(image_shape, output_end)):
        return False

    return True


def select_tiles(tile_shape, output_origin, output_shape):
    ''' Selects the tile indices necessary to populate the resulting image.

    Args:
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image
        output_shape: Tuple of integer height, width of resulting image

    Returns:
        List of tuples of integer y, x tile indices
    '''
    start_yx = get_tile_start(tile_shape, output_origin)
    count_yx = get_tile_count(tile_shape, output_origin, output_shape)

    # Calculate all indices between first and last
    offsets = np.argwhere(np.ones(count_yx - start_yx))
    return list(map(tuple, (start_yx + offsets).tolist()))


def extract_subtile(indices, tile_shape, output_origin, output_shape, tile):
    '''Returns the part of the tile required for the resulting image.

    Args:
        indices: Tuple of integer y, x tile indices
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image
        output_shape: Tuple of integer height, width of resulting image
        tile: 2D integer array for full tile from which to extract

    Returns:
        The part of the tile needed for the resulting image
    '''
    subregion = select_subregion(indices, tile_shape,
                                 output_origin, output_shape)
    # Take subregion from tile
    [yt_0, xt_0], [yt_1, xt_1] = subregion
    return tile[yt_0:yt_1, xt_0:xt_1]


def composite_subtile(out, subtile, position, color, range_min, range_max):
    ''' Composites a subtile into an output image.

    Args:
        out: RBG image float array to contain composited subtile
        subtile: 2D integer subtile needed for the resulting image
        position: Tuple of integer y, x position of tile region
            within the resulting image
        color: Color as r, g, b float array within 0, 1
        range_min: Threshhold range minimum, float within 0, 1
        range_max: Threshhold range maximum, float within 0, 1

    Returns:
        A reference to _out_
    '''
    # Define boundary
    y_0, x_0 = position
    shape = np.int64(subtile.shape)
    y_1, x_1 = [y_0, x_0] + shape

    # Composite the subtile into the output
    composite_channel(out[y_0:y_1, x_0:x_1], subtile,
                      color, range_min, range_max,
                      out[y_0:y_1, x_0:x_1])
    return out


def composite_subtiles(tiles, tile_shape, output_origin, output_shape):
    ''' Positions all image tiles and channels in the resulting image.

    Only the necessary subregions of tiles are combined to produce
    a resulting image matching exactly the size specified.

    Argsr
        tiles: Iterator of tiles to blend. Each dict in the
            iterator must have the following rendering settings:
            {
                indices: Tuple of integer y, x tile indices
                image: Numpy 2D image data of any type for a full tile
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        tile_shape: Tuple of integer height, width of one tile
        output_origin: Tuple of integer y, x origin of resulting image
        output_shape: Tuple of integer height, width of resulting image

    Returns:
        A float32 RGB color image with each channel's shape matching the
        ouptput_shape. Channels contain gamma-corrected values from 0 to 1.
    '''
    output_h, output_w = output_shape
    out = np.zeros((output_h, output_w, 3))

    for tile in tiles:
        idx = tile['indices']
        position = select_position(idx, tile_shape, output_origin)
        subtile = extract_subtile(idx, tile_shape, output_origin,
                                  output_shape, tile['image'])
        composite_subtile(out, subtile, position, tile['color'],
                          tile['min'], tile['max'])

    # Return gamma correct image within 0, 1
    np.clip(out, 0, 1, out=out)
    return ski.adjust_gamma(out, 1 / 2.2)
