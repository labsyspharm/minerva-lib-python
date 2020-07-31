""" Minerva Python Library
"""
import os
from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext
import versioneer

OS_WIN = True if os.name == 'nt' else False

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md')) as f:
    README = f.read()

REQUIRES = [
    'numpy>=1.18',
    'boto3>=1.12.39',
    'requests==2.22.0',
    'scikit-learn',
    'tifffile>=2020.7.22'
]

TEST_REQUIRES = [
    'pytest'
]

# Hack to build "pure CTypes" shared library with setuptools.
#
# https://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
class build_ext(build_ext):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypes)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            lib_extension = '.so' if not OS_WIN else '.dll'
            return ext_name + lib_extension
        return super().get_ext_filename(ext_name)

class CTypes(Extension): pass


GCC_COMPILE_ARGS = ["-std=c99", "-fPIC", "-O3", "-march=haswell", "-ffast-math", "-funsafe-math-optimizations", "-fno-math-errno"]
MSVC_COMPILE_ARGS = ["/O2", "/arch:AVX2"]

COMPILE_ARGS = GCC_COMPILE_ARGS if not OS_WIN else MSVC_COMPILE_ARGS

VERSION = versioneer.get_version()
DESCRIPTION = 'minerva lib'
AUTHOR = 'D.P.W. Russell, Juha Ruokonen'
EMAIL = 'douglas_russell@hms.harvard.edu, juha_ruokonen@hms.harvard.edu'
LICENSE = 'GPL-3.0'
HOMEPAGE = 'https://github.com/sorgerlab/minerva-lib-python'

crender = CTypes('crender',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include'],
                    extra_compile_args=COMPILE_ARGS,
                    sources = ['src/minerva_lib/crender/render.c'])

setup(
    name='minerva-lib',
    version=VERSION,
    cmdclass={'build_ext': build_ext},
    package_dir={'': 'src'},
    description=DESCRIPTION,
    long_description=README,
    packages=find_packages('src'),
    include_package_data=True,
    install_requires=REQUIRES,
    setup_requires=['pytest-runner'],
    tests_require=TEST_REQUIRES,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    url=HOMEPAGE,
    download_url=f'{HOMEPAGE}/archive/v{VERSION}.tar.gz',
    keywords=['minerva', 'library', 'microscopy'],
    zip_safe=False,
    ext_modules=[crender]
)
