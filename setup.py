""" Minerva Python Library
"""
import os
from configparser import ConfigParser
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md')) as f:
    README = f.read()

REQUIRES = [
    'numpy>=1.11.1',
    'scikit-image>=0.13.1'
]

TEST_REQUIRES = [
    'pytest'
]


def read_version():
    """
    Returns:
        Version string of this module
    """
    config = ConfigParser()
    config.read('setup.cfg')
    return config.get('metadata', 'version')


VERSION = read_version()
DESCRIPTION = 'minerva lib'
AUTHOR = 'D.P.W. Russell'
EMAIL = 'douglas_russell@hms.harvard.edu'
LICENSE = 'GPL-3.0'
HOMEPAGE = 'https://github.com/sorgerlab/minerva-lib-python'

setup(
    name='minerva-lib',
    version=VERSION,
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
    download_url='%s/archive/v%s.tar.gz' % (HOMEPAGE, VERSION),
    keywords=['minerva', 'library', 'microscopy'],
    zip_safe=False,
)
