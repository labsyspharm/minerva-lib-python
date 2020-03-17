import itertools
import os, logging, re
from xml.etree import ElementTree

OME_NS = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'

class FileUtils:

    _valid_name = re.compile('^[a-zA-Z][a-zA-Z0-9\\-_]+$')
    _length_name = 128
    _file_pattern = re.compile('C(\\d+)-T(\\d+)-Z(\\d+)-L(\\d+)-Y(\\d+)-X(\\d+).*')

    @staticmethod
    def list_files(dir, filefilter):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if ext in filefilter:
                    files.append(os.path.join(dirpath, filename))

        logging.debug(files)
        return files

    @staticmethod
    def list_files_regex(dir, pattern):
        files = []
        prog = re.compile(pattern)
        for filename in os.listdir(dir):
            if prog.match(filename):
                files.append(os.path.join(dir, filename))
            else:
                logging.info("(skip) Filename %s does not match tile pattern: %s", filename, pattern)

        logging.info(files)
        return files

    @staticmethod
    def validate_name(s, object_type=None):
        if len(s) > FileUtils._length_name or FileUtils._valid_name.match(s) is None:
            raise ValueError('{} name is invalid. Valid names begin with a letter, '
                             'contain only alphanumeric characters, dash and '
                             'underscore. The maximum length '
                             'is {}'.format(object_type, FileUtils._length_name))

    @staticmethod
    def get_key(filename):
        basename = os.path.basename(filename)
        path = os.path.dirname(filename)
        path = path.replace("\\", "/")
        path = re.sub("^[a-zA-Z]:", "", path)
        return path + '/' + basename

    @staticmethod
    def parse_filename(filename):
        m = FileUtils._file_pattern.match(filename)
        if m is None:
            logging.warning("No match for filename %s", filename)

        return {
            "c": m.group(1),
            "t": m.group(2),
            "z": m.group(3),
            "l": m.group(4),
            "y": m.group(5),
            "x": m.group(6)
        }

    @staticmethod
    def transform_xml(metadata: str, image_uuid):
        ElementTree.register_namespace('', OME_NS)
        xml_root = ElementTree.fromstring(metadata)
        iterfind = lambda m: xml_root.iterfind(f'ome:{m}', {'ome': OME_NS})
        # Replace original image ID attributes with our UUID-based IDs.
        for elt in itertools.chain(iterfind('Image'), iterfind('ImageRef')):
            elt.attrib['ID'] = 'Image:' + str(image_uuid)

        xml_str = ElementTree.tostring(xml_root, encoding='utf-8')
        return xml_str

    @staticmethod
    def get_pyramid_levels(filenames):
        max_level = 0
        for filename in filenames:
            m = FileUtils.parse_filename(os.path.basename(filename))
            channel = int(m["c"])
            max_level = max(max_level, channel)

        return max_level + 1


    @staticmethod
    def validate_tiles(files):
        '''
        Validate that files are of supported tile format: 16-bit grayscale png
        '''
        for tile in files:
            with open(tile, 'rb') as f:
                # png signature
                signature = f.read(8)
                png = signature[1:4]
                if png != b'PNG':
                    raise ValueError('Invalid file ' + tile + '. Image must be a PNG image!')
                # signature end

                #  IHDR chunk
                ihdr = f.read(25)
                depth = ihdr[16]
                color = ihdr[17]
                if depth != 16:
                    raise ValueError('Invalid file ' + tile + '. PNG must be 16 bit depth! Depth: ', depth)
                if color != 0:
                    raise ValueError('Invalid file ' + tile + '. PNG must be grayscale! Color: ', color)