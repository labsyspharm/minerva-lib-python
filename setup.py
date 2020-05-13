""" Minerva Python Library
"""
import os
from setuptools import setup, find_packages, Extension
import versioneer


HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md')) as f:
    README = f.read()

REQUIRES = [
    'numpy>=1.11.1',
    'boto3==1.11.12',
    'requests==2.22.0'
]

TEST_REQUIRES = [
    'pytest'
]

COMPILE_ARGS = ["-std=c99", "-fPIC", "-mmmx", "-msse", "-msse2", "-msse3", "-mssse3", "-msse4", "-mavx", "-mavx2", "-O3"]

VERSION = versioneer.get_version()
DESCRIPTION = 'minerva lib'
AUTHOR = 'D.P.W. Russell, Juha Ruokonen'
EMAIL = 'douglas_russell@hms.harvard.edu, juha_ruokonen@hms.harvard.edu'
LICENSE = 'GPL-3.0'
HOMEPAGE = 'https://github.com/sorgerlab/minerva-lib-python'

crender = Extension('crender',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include'],
                    extra_compile_args=COMPILE_ARGS,
                    sources = ['src/minerva_lib/crender/render.c'])

setup(
    name='minerva-lib',
    version=VERSION,
    cmdclass=versioneer.get_cmdclass(),
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
