import io
import logging, time, sys, os
import random, string, re
import math
from concurrent.futures import ThreadPoolExecutor
import zarr

from .client import MinervaClient
from .util.progress import ProgressPercentage
from .util.s3 import S3Uploader
from .util.fileutils import FileUtils
from io import BytesIO
from tifffile import TiffFile
import tifffile
import uuid

logger = logging.getLogger("minerva")

class TileData:
    def __init__(self, data: BytesIO, channel=0, time=0, z=0, level=0, y=0, x=0, extension="png"):
        self.data = data
        self.channel = channel
        self.time = time
        self.z = z
        self.level = level
        self.y = y
        self.x = x
        self.extension = extension
        TileData._check_positive([self.channel, self.time, self.z, self.level, self.y, self.x])

    def get_filename(self):
        return "C{}-T{}-Z{}-L{}-Y{}-X{}.{}".format(self.channel, self.time, self.z, self.level, self.y, self.x, self.extension)

    @staticmethod
    def _check_positive(values):
        for value in values:
            if value < 0:
                raise ValueError("Values must be positive!")

class MinervaImporter:

    def __init__(self, minerva_client: MinervaClient, uploader: S3Uploader, region="us-east-1", dryrun=False):
        self.minerva_client = minerva_client
        self.uploader = uploader
        self.region = region
        self.executor = ThreadPoolExecutor()
        self.dryrun = dryrun

    def import_files(self, files, repository=None, archive=False):
        repository_uuid = self._create_or_get_repository(repository, archive)
        # Create a random name for import
        import_uuid = self._create_import(repository_uuid)
        logger.info("Created new import, uuid: %s", import_uuid)
        # Get AWS credentials for S3 bucket for raw image
        credentials, bucket, prefix = self._get_import_credentials(import_uuid)
        logger.info("Uploading to S3 bucket: %s/%s", bucket, prefix)

        # Upload all files in parallel to S3
        self._upload_raw_files(files, bucket, prefix, credentials)

        self.minerva_client.mark_import_complete(import_uuid)
        return import_uuid

    def create_image(self, name, repository, format, compression=None, pyramid_levels=1, tile_size=1024):
        if self.dryrun:
            return uuid.uuid4()

        repository_uuid = self._create_or_get_repository(repository, False)
        if name is None:
            name = 'IMG_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))

        res = self.minerva_client.create_image(name, repository_uuid, format, compression=compression, pyramid_levels=pyramid_levels, tile_size=tile_size)
        image_uuid = res["data"]["uuid"]
        logger.info("Created new image, uuid: %s", image_uuid)
        return image_uuid

    def direct_import(self, tile_data: TileData, image_uuid, async_upload=False):
        if not isinstance(tile_data, TileData):
            raise ValueError('First argument must be instance of TileData!')

        credentials, bucket, prefix = self._get_image_credentials(image_uuid)
        logger.debug("Credentials %s", credentials)
        logger.debug("Bucket %s", bucket)
        logger.debug("Prefix %s", prefix)
        key = prefix + "/" + tile_data.get_filename()
        if not async_upload:
            self.uploader.upload_data(tile_data.data, bucket, key, credentials)
        else:
            self.executor.submit(self.uploader.upload_data, tile_data.data, bucket, key, credentials)

    def direct_import_files(self, files, image_uuid, async_upload=False):
        credentials, bucket, prefix = self._get_image_credentials(image_uuid)
        logger.debug("Credentials %s", credentials)
        logger.debug("Bucket %s", bucket)
        logger.debug("Prefix %s", prefix)

        for file in files:
            key = prefix + "/" + os.path.basename(file)
            if not async_upload:
                self.uploader.upload_file(file, bucket, key, credentials)
            else:
                self.executor.submit(self.uploader.upload_file, file, bucket, key, credentials)

    def direct_import_metadata(self, metadata, image_uuid):
        if self.dryrun:
            return

        logger.info("Importing metadata for image %s", str(image_uuid))
        xml = FileUtils.transform_xml(metadata, image_uuid)
        credentials, bucket, prefix = self._get_image_credentials(image_uuid)
        logger.debug("Credentials %s", credentials)
        logger.debug("Bucket %s", bucket)
        logger.debug("Prefix %s", prefix)
        self.uploader.upload_data(xml, bucket, prefix + '/metadata.xml', credentials)

    def wait_upload(self):
        self.executor.shutdown(wait=True)

    def _create_or_get_repository(self, repository, archive=False):
        res = self.minerva_client.list_repositories()
        existing_repository = list(filter(lambda x: x["name"] == repository, res["included"]["repositories"]))
        if len(existing_repository) == 0:
            raw_storage = "Destroy" if not archive else "Archive"
            res = self.minerva_client.create_repository(repository, raw_storage=raw_storage)
            repository_uuid = res["data"]["uuid"]
            logger.info("Created new repository, uuid: %s", repository_uuid)
        else:
            repository_uuid = existing_repository[0]["uuid"]
            logger.info("Using existing repository uuid: %s", repository_uuid)
            if archive:
                logger.warning("Archive flag has no effect, because using existing repository!")
        return repository_uuid

    def _create_import(self, repository_uuid):
        import_name = 'I' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
        res = self.minerva_client.create_import(import_name, repository_uuid)
        return res["data"]["uuid"]

    def _get_import_credentials(self, import_uuid):
        res = self.minerva_client.get_import_credentials(import_uuid)
        m = re.match(r"^s3://([A-z0-9\-]+)/([A-z0-9\-]+)/$", res["data"]["url"])
        bucket = m.group(1)
        prefix = m.group(2)
        credentials = res["data"]["credentials"]
        return credentials, bucket, prefix

    def _get_image_credentials(self, image_uuid):
        if self.dryrun:
            return ({}, "bucket", "prefix")

        return self.minerva_client.get_image_credentials(image_uuid)

    def _upload_raw_files(self, files, bucket, prefix, credentials):
        progress = ProgressPercentage()
        for file in files:
            progress._total_size += os.path.getsize(file)

        with ThreadPoolExecutor() as executor:
            for file in files:
                key = prefix + FileUtils.get_key(file)
                executor.submit(self.uploader.upload_file, file, bucket, key, credentials, progress)

        sys.stdout.write("\r\n")

    def poll_import_progress(self, import_uuid):
        all_complete = False
        timeout = 1800  # 30 mins (initial timeout for EFS syncing)
        extract_timeout = 1800  # 30 mins (extend timeout when bfextraction starts)
        timeout_extended = False
        start = time.time()
        logger.info("Please wait while filesets are created...")
        while not all_complete:
            result = self.minerva_client.list_filesets_in_import(import_uuid)
            filesets = result["data"]
            if len(filesets) > 0:
                if not timeout_extended:
                    timeout_extended = True
                    timeout = timeout + extract_timeout
                all_complete = True
                progresses = []
                for fileset in filesets:
                    all_complete = all_complete and fileset["complete"]
                    progress = fileset["progress"] if fileset["progress"] is not None else 0
                    progresses.append((fileset, progress))

                MinervaImporter._print_progress(progresses)

            if not all_complete:
                time_spent = time.time() - start
                if time_spent > timeout:
                    logger.warning("Waiting for import timed out!")
                    logger.warning("This does not necessarily mean that import failed, it could just take longer than expected.")
                    logger.warning("To check fileset progress, run command:")
                    logger.warning("minerva status")
                    all_complete = True

                time.sleep(2)

    @staticmethod
    def _print_progress(progresses):
        if len(progresses) == 0:
            return

        sys.stdout.write("\rProcessing filesets: ")
        for p in progresses:
            fileset = p[0]
            progress = p[1]
            sys.stdout.write("{} {}% ".format(fileset["name"], progress))

    def _calculate_total_tiles(self, group):
        total_tiles = 0
        for level in group:
            img = group[level]
            channels = img.shape[0]
            height = img.shape[1]
            width = img.shape[2]
            total_tiles += channels * math.ceil(height / 1024) * math.ceil(width / 1024)

        return total_tiles

    def import_ome_tiff(self, file, repository, tile_size=1024, progress_callback=lambda a,b : None, filename=None):
        if filename is None:
            filename = os.path.basename(file)

        with TiffFile(file, is_ome=False) as tif:
            group = zarr.open(tif.series[0].aszarr())

            if len(group) <= 1:
                raise ValueError("Image does not contain pyramid levels.")

            total_tiles = self._calculate_total_tiles(group)
            image_uuid = self.create_image(filename, repository, format="tiff", compression="zstd", pyramid_levels=len(group), tile_size=tile_size)
            credentials, bucket, prefix = self._get_image_credentials(image_uuid)
            metadata = tif.pages[0].tags['ImageDescription'].value
            self.direct_import_metadata(metadata, image_uuid)

            tiles_processed = 0

            def done_callback(f):
                progress_callback(tiles_processed, total_tiles)

            for pyramid_level in group:
                img = group[pyramid_level]

                num_channels = img.shape[0]
                tiles_height = math.ceil(img.shape[1] / tile_size)
                tiles_width = math.ceil(img.shape[2] / tile_size)

                # TODO - Handle t and z dimensions
                t = 0
                z = 0
                for channel in range(num_channels):
                    for tile_x in range(0, tiles_width):
                        for tile_y in range(0, tiles_height):
                            logger.debug("Processing L=%s C=%s X=%s Y=%s", pyramid_level, channel, tile_x, tile_y)
                            tile = img[channel, tile_y*tile_size:(tile_y+1)*tile_size, tile_x*tile_size:(tile_x+1)*tile_size]

                            buf = io.BytesIO()
                            tifffile.imwrite(buf, tile, compress=("ZSTD", 1))
                            buf.seek(0)
                            filename = f'C{channel}-T{t}-Z{z}-L{pyramid_level}-Y{tile_y}-X{tile_x}.tif'
                            tile_key = f'{prefix}/{filename}'

                            if not self.dryrun:
                                future = self.uploader.async_upload(buf, bucket, tile_key, credentials)
                                future.add_done_callback(done_callback)
                            else:
                                progress_callback(tiles_processed, total_tiles)

                            tiles_processed += 1

        self.uploader.wait_upload()
