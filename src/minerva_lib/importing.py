import io
import logging, time, sys, os
import random, string, re
import math
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import zarr
import s3fs
import boto3
from tifffile import TiffFile
import itertools

from .client import MinervaClient
from .util.progress import ProgressPercentage
from .util.s3 import S3Uploader
from .util.fileutils import FileUtils
from io import BytesIO
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
        self.executor = ThreadPoolExecutor(max_workers=10)
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

    def direct_import_metadata(self, metadata, image_uuid, credentials=None, bucket=None, prefix=None):
        if self.dryrun:
            return

        logger.info("Importing metadata for image %s", str(image_uuid))
        xml = FileUtils.transform_xml(metadata, image_uuid)
        if credentials is None or bucket is None or prefix is None:
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
            return ({"AccessKeyId": "", "SecretAccessKey": "",  "SessionToken": ""}, "bucket", "prefix")
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

    def _get_dimensions(self, level):
        if len(level.shape) == 3:
            channels = level.shape[0]
            height = level.shape[1]
            width = level.shape[2]
        else:
            channels = 1
            height = level.shape[0]
            width = level.shape[1]

        return channels, height, width

    def _calculate_total_tiles(self, group):
        total_tiles = 0
        for name in group:
            level = group[name]
            channels, height, width = self._get_dimensions(level)
            total_tiles += channels * math.ceil(height / 1024) * math.ceil(width / 1024)

        return total_tiles

    def import_ome_tiff(self, file, repository, tile_size=1024, progress_callback=lambda a,b : None, image_name=None):
        """
        Processes an ome.tif client side and imports it directly into S3 tilebucket.

        Parameters
        ----------
        file - File path
        repository - Repository name
        tile_size - Tile size, default 1024
        progress_callback - Callback function to report progress
        image_name - Image name, by default is taken from the filename
        """
        if image_name is None:
            image_name = os.path.basename(file)

        executor = ThreadPoolExecutor(max_workers=10)
        # limit the queue of pending uploads to 100
        queue_limit = 100
        futures = set()

        with TiffFile(file, is_ome=False) as tif:
            # Depending on whether the image contains pyramid or not,
            # this will either be Zarr Group or Array
            group_or_array = zarr.open(tif.aszarr())

            if isinstance(group_or_array, zarr.core.Array):
                i = 0 if len(group_or_array.shape) == 2 else 1
                if group_or_array.shape[i] > tile_size and group_or_array.shape[i+1] > tile_size:
                    logger.error("Local importing of images without pyramid is not currently supported. Use server-side importing instead.")
                    raise ValueError("Image is larger than TILE_SIZE but does not contain pyramid levels.")
                num_levels = 1
            else:
                num_levels = len(group_or_array)

            image_uuid = self.create_image(image_name,
                                           repository,
                                           format="zarr",
                                           compression="zstd",
                                           pyramid_levels=num_levels,
                                           tile_size=tile_size)

            credentials, bucket, prefix = self._get_image_credentials(image_uuid)

            metadata = tif.pages[0].tags['ImageDescription'].value
            self.direct_import_metadata(metadata,
                                        image_uuid,
                                        credentials=credentials,
                                        bucket=bucket,
                                        prefix=prefix)

            total_tiles = self._calculate_total_tiles(group_or_array)
            tiles_processed = 0
            def done_callback(f):
                progress_callback(tiles_processed, total_tiles)

            compressor = zarr.Blosc(cname='zstd', clevel=3)

            s3 = s3fs.S3FileSystem(anon=self.dryrun,
                                   client_kwargs=dict(region_name=self.region),
                                   key=credentials["AccessKeyId"],
                                   secret=credentials["SecretAccessKey"],
                                   token=credentials["SessionToken"])

            if not self.dryrun:
                zarr_store = s3fs.S3Map(root=f"{bucket}/{prefix}", s3=s3, check=False, create=False)
            else:
                zarr_store = zarr.DirectoryStore("./zarrtmp")

            # In OME-ZARR each pyramid level will be stored as a separate zarr Array, named by
            # the index number of the level, e.g. "0" is highest detail level
            # All Arrays are stored under a zarr Group.
            output = zarr.group(store=zarr_store, overwrite=True)

            for pyramid_level in range(num_levels):
                if isinstance(group_or_array, zarr.core.Array):
                    img = group_or_array
                else:
                    img = group_or_array[pyramid_level]

                num_channels, height, width = self._get_dimensions(img)

                tiles_height = math.ceil(img.shape[1] / tile_size)
                tiles_width = math.ceil(img.shape[2] / tile_size)

                arr = output.create(shape=(1, num_channels, 1, height, width), chunks=(1, 1, 1, 1024, 1024),
                                    name=str(pyramid_level), dtype=img.dtype, compressor=compressor)

                # TODO - Handle t and z dimensions
                t = 0
                z = 0
                channels_range = range(num_channels)
                x_range = range(0, tiles_width)
                y_range = range(0, tiles_height)

                for channel, tile_x, tile_y in itertools.product(channels_range, x_range, y_range):
                    logger.debug("Processing L=%s C=%s X=%s Y=%s", pyramid_level, channel, tile_x, tile_y)
                    x = tile_x * tile_size
                    y = tile_y * tile_size
                    tile = img[channel, y:y + tile_size, x:x + tile_size]

                    if len(futures) >= queue_limit:
                        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    future = executor.submit(self._upload_zarr, arr, t, channel, z, y, x, tile_size, tile)
                    futures.add(future)

                    progress_callback(tiles_processed, total_tiles)
                    tiles_processed += 1

        executor.shutdown()

    def _upload_zarr(self, arr, t, channel, z, y, x, tile_size, tile):
        arr[t, channel, z, y:y + tile_size, x:x + tile_size] = tile
