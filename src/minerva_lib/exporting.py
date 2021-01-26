from .client import MinervaClient
import logging
import math
import itertools
import tifffile
import numpy as np
import boto3
import os
from concurrent.futures import ThreadPoolExecutor
from time import time
from requests.exceptions import HTTPError

SOFTWARE_TAG_CODE = 305

logger = logging.getLogger("minerva")

class MinervaExporter:
    def __init__(self, region):
        self.region = region

    def export_image(self, minerva_client: MinervaClient, image_uuid: str, output_path: str, save_pyramid=False, progress_callback=lambda a,b : None, format="zarr"):
        image, ome_metadata = self._get_image_and_metadata(minerva_client, image_uuid)
        if image is None:
            raise KeyError(image_uuid)
        logger.debug(ome_metadata)

        if format == "zarr":
            return self.export_image_zarr(image, ome_metadata, minerva_client, image_uuid, output_path, save_pyramid, progress_callback)
        else:
            return self.export_image_ometiff(image, ome_metadata, minerva_client, image_uuid, output_path, save_pyramid, progress_callback)

    def export_image_zarr(self, image, ome_metadata, minerva_client: MinervaClient, image_uuid: str, output_path: str, save_pyramid=False, progress_callback=lambda a,b : None):
        credentials, bucket, prefix = minerva_client.get_image_credentials(image_uuid)
        s3 = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                          aws_secret_access_key=credentials["SecretAccessKey"],
                          aws_session_token=credentials["SessionToken"],
                          region_name=self.region)

        objs = []
        more_objects = True
        continuation_token = None
        while more_objects:
            args = {
                "Bucket": bucket,
                "Prefix": image_uuid,
                "MaxKeys": 10000,
            }
            if continuation_token is not None:
                args["ContinuationToken"] = continuation_token

            result = s3.list_objects_v2(**args)
            objs.extend(result["Contents"])
            if not result["IsTruncated"]:
                more_objects = False
            else:
                continuation_token = result["NextContinuationToken"]

        total_files = len(objs)
        files_processed = 0
        def done_callback(f):
            nonlocal files_processed
            nonlocal total_files
            files_processed += 1
            progress_callback(files_processed, total_files)

        executor = ThreadPoolExecutor(max_workers=10)
        for obj in objs:
            key = obj["Key"]
            logger.debug("Downloading key %s", key)
            path = os.path.join(output_path, key)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            future = executor.submit(self._s3_download_file, credentials, bucket, key, str(path), self.region)
            future.add_done_callback(done_callback)

        executor.shutdown(wait=True)

    def _s3_download_file(self, credentials, bucket, key, filename, region):
        s3 = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                          aws_secret_access_key=credentials["SecretAccessKey"],
                          aws_session_token=credentials["SessionToken"],
                          region_name=self.region)
        s3.download_file(Bucket=bucket, Key=key, Filename=filename)

    # TODO TEST OME-TIFF EXPORT!
    def export_image_ometiff(self, image, ome_metadata, minerva_client: MinervaClient, image_uuid: str, output_path: str, save_pyramid=False, progress_callback=lambda a,b : None):
        start = time()
        image, ome_metadata = self._get_image_and_metadata(minerva_client, image_uuid)
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
                        f = executor.submit(self._download_tile, minerva_client, img_level, image_uuid, x, y, 0, 0, channel, level, tile_size)
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

    def _get_image_and_metadata(self, minerva_client, image_uuid):
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

    def _download_tile(self, minerva_client, arr, image_uuid, x, y, z, t, channel, level, tile_size):
        tile = minerva_client.get_raw_tile(image_uuid, x, y, z, t, channel, level)
        arr[y * tile_size:y * tile_size + tile.shape[0], x * tile_size:x * tile_size + tile.shape[1]] = tile
