from .client import MinervaClient
import logging
import math
import itertools
import tifffile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from time import time
from requests.exceptions import HTTPError

SOFTWARE_TAG_CODE = 305

logger = logging.getLogger("minerva")

def export_image(minerva_client: MinervaClient, image_uuid: str, output_path: str, save_pyramid=False, progress_callback=lambda a,b : None):
    start = time()
    image, ome_metadata = _get_image_and_metadata(minerva_client, image_uuid)
    if image is None:
        raise KeyError(image_uuid)
    logger.debug(ome_metadata)

    if output_path is None:
        output_path = image["included"]["images"][0]["name"]
        if output_path.endswith(".ome"):
            output_path += ".tif"
        elif not output_path.endswith(".ome.tif"):
            output_path += ".ome.tif"

    tiles_processed = 0
    total_tiles = 0

    def done_callback(f):
        progress_callback(tiles_processed, total_tiles)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        num_channels = len(image["data"]["pixels"]["channels"])
        pyramid_levels = image["included"]["images"][0]["pyramid_levels"] if save_pyramid else 1
        tile_size = image["included"]["images"][0]["tile_size"]
        width = image["data"]["pixels"]["SizeX"]
        height = image["data"]["pixels"]["SizeY"]
        for level in range(pyramid_levels):
            logger.debug("Pyramid level %s/%s", level, pyramid_levels-1)
            tiles_width = math.ceil(width / tile_size)
            tiles_height = math.ceil(height / tile_size)
            total_tiles += (tiles_width*tiles_height*num_channels)

            for channel in range(num_channels):
                logger.debug("Fetch channel %s/%s", channel, num_channels-1)

                img_level = np.zeros(shape=(height, width), dtype=np.uint16)
                executor = ThreadPoolExecutor(max_workers=10)
                for x, y in itertools.product(range(tiles_width), range(tiles_height)):
                    f = executor.submit(_download_tile, minerva_client, img_level, image_uuid, x, y, 0, 0, channel, level, ".tif", tile_size)
                    f.add_done_callback(done_callback)

                subfiletype = 0 if (level == 0) else 1
                extra_tags = [(SOFTWARE_TAG_CODE, "s", 1, "Minerva (Glencoe/Faas pyramid output)", True)]
                options = dict(tile=(1024, 1024))
                # Write metadata to first page only
                description = ome_metadata if (channel == 0 and level == 0) else None

                executor.shutdown(wait=True)
                tif.save(img_level, metadata=None, contiguous=False, subfiletype=subfiletype, description=description, extratags=extra_tags, **options)

            width = math.ceil(width / 2)
            height = math.ceil(height / 2)

    logger.debug("Completed - export time: %s", time() - start)
    logger.debug("Image file: %s", output_path)
    return output_path

def _get_image_and_metadata(minerva_client, image_uuid):
    try:
        image = minerva_client.get_image_dimensions(image_uuid)
        ome_metadata = minerva_client.get_image_metadata(image_uuid)
        return image, ome_metadata
    except HTTPError as e:
        if e.response.status_code == 403:
            logger.error("Image %s does not exist or insufficient permissions", image_uuid)
            return None, None
        elif e.response.status_code == 422:
            logger.error("%s is not a valid UUID", image_uuid)
            return None, None
        raise e

def _download_tile(minerva_client, arr, image_uuid, x, y, z, t, channel, level, format, tile_size):
    tile = minerva_client.get_raw_tile(image_uuid, x, y, z, t, channel, level, format=".tif")
    arr[y * tile_size:y * tile_size + tile.shape[0], x * tile_size:x * tile_size + tile.shape[1]] = tile
