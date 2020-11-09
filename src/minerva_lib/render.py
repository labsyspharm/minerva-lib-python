import itertools
import collections.abc
import numpy as np
from . import skimage_inline as ski
from .crender.wrapper import crender, c_uint8_p, c_uint16_p, c_uint32_p, c_uint64_p

def composite_channel(target, image, color, range_min, range_max, out=None):
    ''' Render _image_ in pseudocolor and composite into _target_

    By default, a new output array will be allocated to hold
    the result of the composition operation. To update _target_
    in place instead, specify the same array for _target_ and _out_.

    Args:
        target: Numpy array containing composition target image
        image: Numpy array of image to render and composite
            !! numpy views are NOT handled correctly by the native rendering code
            !! If you want to use a view, make a copy first
        color: Color as r, g, b float array within 0, 1
        range_min: Threshhold range minimum, float within 0, 1
        range_max: Threshhold range maximum, float within 0, 1
        out: Optional output numpy array in which to place the result.

    Returns:
        A numpy array with the same shape as the composited image.
        If an output array is specified, a reference to _out_ is returned.
    '''
    if out is None:
        out = target.copy()

    length = target.shape[0] * target.shape[1]

    if image.dtype == 'uint32':
        image_p = image.ctypes.data_as(c_uint32_p)
        crender.rescale_intensity32(image_p, int(range_min*(2**32)), int(range_max*(2**32)), length)
        out_p = out.ctypes.data_as(c_uint64_p)
        crender.composite32(out_p, image_p, color[0], color[1], color[2], length)

    elif image.dtype == 'uint16':
        image_p = image.ctypes.data_as(c_uint16_p)
        crender.rescale_intensity16(image_p, int(range_min*65535), int(range_max*65535), length)
        out_p = out.ctypes.data_as(c_uint32_p)
        crender.composite16(out_p, image_p, color[0], color[1], color[2], length)

    elif image.dtype == 'uint8':
        image_p = image.ctypes.data_as(c_uint8_p)
        crender.rescale_intensity8(image_p, int(range_min*255), int(range_max*255), length)
        out_p = out.ctypes.data_as(c_uint16_p)
        crender.composite8(out_p, image_p, color[0], color[1], color[2], length)
    return out

def composite_channels(channels, gamma=None):
    '''Render each image in _channels_ additively into a composited image

    Args:
        channels: List of dicts for channels to blend. Each dict in the
            list must have the following rendering settings:
            {
                image: Numpy 2D image data of any type
                    !! numpy views are NOT handled correctly by the native rendering code
                    !! If you want to use a view, make a copy first
                color: Color as r, g, b float array within 0, 1
                min: Threshhold range minimum, float within 0, 1
                max: Threshhold range maximum, float within 0, 1
            }
        gamma: Gamma correction value, default 1/2.2 (1 = no gamma)

    Returns:
        For input images with shape `(n,m)`,
        returns a uint8 RGB color image with shape
        `(n,m,3)` and values in the range 0 to 255
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
        if not channel['image'].flags['OWNDATA']:
            print("WARNING: Image is possibly a numpy view, views are not rendered correctly!")

    source_dtype = channels[0]['image'].dtype
    # Shape of 3 color image
    shape_color = shape + (3,)

    # Final buffer for blending
    if source_dtype == 'uint16':
        out_buffer = np.zeros(shape_color, dtype=np.uint32)
    elif source_dtype == 'uint32':
        out_buffer = np.zeros(shape_color, dtype=np.uint64)
    elif source_dtype == 'uint8':
        out_buffer = np.zeros(shape_color, dtype=np.uint16)

    # rescaled images and normalized colors
    for channel in channels:
        # Add all three channels to output buffer
        args = map(channel.get, ['image', 'color', 'min', 'max'])
        composite_channel(out_buffer, *args, out=out_buffer)

    out8 = np.empty(shape_color, dtype=np.uint8)
    out8_p = out8.ctypes.data_as(c_uint8_p)

    if source_dtype == 'uint16':
        out_buffer_p = out_buffer.ctypes.data_as(c_uint32_p)
        crender.clip32_conv8(out_buffer_p, out8_p, shape[0]*shape[1]*3)
    elif source_dtype == 'uint32':
        out_buffer_p = out_buffer.ctypes.data_as(c_uint64_p)
        crender.clip64_conv8(out_buffer_p, out8_p, shape[0]*shape[1]*3)
    elif source_dtype == 'uint8':
        out_buffer_p = out_buffer.ctypes.data_as(c_uint16_p)
        crender.clip16_conv8(out_buffer_p, out8_p, shape[0]*shape[1]*3)

    # Return gamma correct image within 0, 1
    if gamma is None:
        # Default gamma value if no parameter is given
        gamma = 1 / 2.2

    if gamma != 1:
        ski.adjust_gamma(out8, gamma)

    return out8


def scale_image_nearest_neighbor(source, factors):
    '''Resizes an image by the given factors using nearest neighbor pixels.

    Args:
        source: A 2D grayscale or RGB numpy array to resize.
        factors: Tuple of height, width float ratios of output image shape to
            input shape, or a single ratio to be used for both height and
            width.

    Returns:
        A numpy array with the resized source image.
    '''

    if isinstance(factors, collections.abc.Collection):
        if len(factors) != 2:
            raise ValueError('Factors must be a 2-tuple or a single value')
        factors = tuple(factors)
    else:
        factors = (factors, factors)
    if any(f <= 0 for f in factors):
        raise ValueError('Factors must all be positive')

    factors += (1,)

    # The output will have the same number of color channels as the source
    s_lim = [d - 1 for d in source.shape]
    o_shape = [int(round(s * f)) for s, f in zip(source.shape, factors)]

    x_index = np.round(np.linspace(0, s_lim[1], o_shape[1])).astype(int)
    y_index = np.round(np.linspace(0, s_lim[0], o_shape[0])).astype(int)

    output = source[y_index][:, x_index]

    return output


def get_optimum_pyramid_level(input_shape, level_count,
                              output_size, prefer_higher_resolution):
    '''Return optimum pyramid level.

    Calculates the pyramid level which most closely matches (above or below,
    depending on `prefer_higher_resolution`) `output_size`, the maximum
    dimension of an output image.

    Args:
        input_shape: Tuple of integer height, width at full resolution.
        level_count: Integer number of available pyramid levels.
        output_size: Integer length of output image longest dimension.
        prefer_higher_resolution: Set True to calculate the pyramid level for a
            resolution exceeding or equal to the ideal resolution. Set False to
            calculate the pyramid level for a resolution less than or equal to
            the ideal resolution.

    Returns:
        Integer pyramid level.
    '''

    longest_side = max(*input_shape)
    ratio_log = np.log2(longest_side / output_size)

    if prefer_higher_resolution:
        level = np.floor(ratio_log)
    else:
        level = np.ceil(ratio_log)

    return int(np.clip(level, 0, level_count - 1))


def transform_coordinates_to_level(coordinates, level):
    '''Transform coordinates from full image space to pyramid level space.

    Args:
        coordinates: Tuple of integer coordinates to transform.
        level: Integer pyramid level.

    Returns:
        Tuple of transformed integer coordinates.
    '''

    scaled_coords = np.array(coordinates) / (2 ** level)
    integer_coords = np.int64(np.round(scaled_coords))
    return tuple(integer_coords.tolist())


def get_region_first_grid(tile_shape, region_origin):
    '''Return the indices of the first tile in the region.

    Args:
        tile_shape: Tuple of integer height, width of one tile.
        region_origin: Tuple of integer y, x origin of region.

    Returns:
        Two-item int64 array of y, x tile tile grid reference.
    '''

    start = np.array(region_origin)

    return np.int64(np.floor(start / tile_shape))


def get_region_grid_shape(tile_shape, region_origin, region_shape):
    '''Return number of tiles along height, width of a region.

    Args:
        tile_shape: Tuple of integer height, width of one tile.
        region_origin: Tuple of integer y, x origin of region.
        region_shape: Tuple of integer height, width of region.

    Returns:
        Two-item int64 array tile count along height, width.
    '''

    first_tile = get_region_first_grid(tile_shape, region_origin)
    last_tile = np.array(region_origin) + region_shape
    shape = last_tile - first_tile

    return np.int64(np.ceil(shape / tile_shape))


def select_subregion(grid, tile_shape, output_origin, output_shape):
    '''Determines the subregion of a tile required for the output image.

    Args:
        grid: Tuple of integer y, x tile grid reference.
        tile_shape: Tuple of integer height, width of one tile.
        output_origin: Tuple of integer y, x origin of output image.
        output_shape: Tuple of integer height, width of output image.

    Returns:
        Start y, x and end y, x integer pixel coordinates for the part of the
        tile needed for the output image.
    '''

    tile_start = np.int64(grid) * tile_shape
    output_end = np.int64(output_origin) + output_shape
    tile_end = tile_start + tile_shape

    # At the start of the tile or the start of the requested region
    y_0, x_0 = np.maximum(output_origin, tile_start) - tile_start
    # At the end of the tile or the end of the requested region
    y_1, x_1 = np.minimum(tile_end, output_end) - tile_start

    return (y_0, x_0), (y_1, x_1)


def select_position(grid, tile_shape, output_origin):
    '''Determines where in the output image to insert the subregion.

    Args:
        grid: Tuple of integer y, x tile grid reference.
        tile_shape: Tuple of integer height, width of one tile.
        output_origin: Tuple of integer y, x origin of output image.

    Returns:
        Tuple of integer y, x position of tile region within output image.
    '''

    tile_start = np.int64(grid) * tile_shape

    # At the start of the tile or the start of the requested region
    y_0, x_0 = np.maximum(output_origin, tile_start) - output_origin

    return (y_0, x_0)


def validate_region_bounds(output_origin, output_shape, image_shape):
    '''Returns True if output image coordinates are within full image.

    Args:
        output_origin: Tuple of integer y, x origin of output image.
        output_shape: Tuple of integer height, width of output image.
        image_shape: Tuple of integer height, width of full image.

    Returns:
        True if output image coordinates are within full image.
    '''

    # Convert to numpy for vector operations
    output_origin = np.array(output_origin)
    output_shape = np.array(output_shape)
    image_shape = np.array(image_shape)

    return (
        # Are the output image dimensions non-zero
        all(output_shape > 0)
        # Is the output origin within the image
        and all(output_origin >= 0)
        # Is the output extent within the image
        and all(output_origin + output_shape <= image_shape)
    )


def select_grids(tile_shape, output_origin, output_shape):
    '''Selects the tile grid references required for the output image.

    Args:
        tile_shape: Tuple of integer height, width of one tile.
        output_origin: Tuple of integer y, x origin of output image.
        output_shape: Tuple of integer height, width of output image.

    Returns:
        List of tuples of integer y, x tile grid references.
    '''

    start_yx = get_region_first_grid(tile_shape, output_origin)
    count_yx = get_region_grid_shape(tile_shape, output_origin, output_shape)

    # Calculate all tile grid references between first and last
    return list(itertools.product(
        range(start_yx[0], start_yx[0] + count_yx[0]),
        range(start_yx[1], start_yx[1] + count_yx[1])
    ))


def extract_subtile(grid, tile_shape, output_origin, output_shape, tile):
    '''Returns the part of the tile required for the output image.

    Args:
        grid: Tuple of integer y, x tile grid reference.
        tile_shape: Tuple of integer height, width of one tile.
        output_origin: Tuple of integer y, x origin of output image.
        output_shape: Tuple of integer height, width of output image.
        tile: Full tile image from which to extract.

    Returns:
        The subregion of the tile needed for the output image.
    '''

    subregion = select_subregion(grid, tile_shape, output_origin,
                                 output_shape)
    # Take subregion from tile
    [yt_0, xt_0], [yt_1, xt_1] = subregion
    return tile[yt_0:yt_1, xt_0:xt_1]


def composite_subtile(out, subtile, position, color, range_min, range_max):
    '''Composites a subtile into an output image.

    Args:
        out: RBG image float array to contain composited subtile.
        subtile: 2D integer subtile needed for the output image.
        position: Tuple of integer y, x position of tile region
            within the output image.
        color: Color as r, g, b float array within 0, 1.
        range_min: Threshold range minimum, float within 0, 1.
        range_max: Threshold range maximum, float within 0, 1.

    Returns:
        A reference to `out`.
    '''

    # Define boundary
    y_0, x_0 = position
    shape = np.int64(subtile.shape)
    y_1, x_1 = [y_0, x_0] + shape

    # Composite the subtile into the output
    composite_channel(out[y_0:y_1, x_0:x_1], subtile, color, range_min,
                      range_max, out[y_0:y_1, x_0:x_1])
    return out


def composite_subtiles(tiles, tile_shape, output_origin, output_shape,
                       target_gamma=2.2):
    '''Positions all image tiles and channels in the output image.

    Only the necessary subregions of tiles are combined to produce a output
    image matching exactly the size specified.

    Args:
        tiles: Iterator of tiles to blend. Each dict in the
            iterator must have the following rendering settings:
            {
                grid: Tuple of integer y, x tile grid reference
                image: Numpy 2D image data of any type for a full tile
                color: Color as r, g, b float array within 0, 1
                min: Threshold range minimum, float within 0, 1
                max: Threshold range maximum, float within 0, 1
            }
        tile_shape: Tuple of integer height, width of one tile.
        output_origin: Tuple of integer y, x origin of output image.
        output_shape: Tuple of integer height, width of output image.
        target_gamma: Gamma of expected output device. Defaults to 2.2.

    Returns:
        A float32 RGB color image with each channel's shape matching the
        `output_shape`. Channels contain gamma-corrected values from 0 to 1.
    '''

    output_h, output_w = output_shape
    out = np.zeros((output_h, output_w, 3))

    for tile in tiles:
        idx = tile['grid']
        position = select_position(idx, tile_shape, output_origin)
        subtile = extract_subtile(idx, tile_shape, output_origin, output_shape,
                                  tile['image'])
        composite_subtile(out, subtile, position, tile['color'], tile['min'],
                          tile['max'])

    # Return gamma correct image within 0, 1
    np.clip(out, 0, 1, out=out)
    return ski.adjust_gamma(out, 1 / target_gamma)
