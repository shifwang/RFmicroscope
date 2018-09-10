#! /usr/bin/env python

import sys
import os
import setuptools
import numpy
from numpy.distutils.core import setup
from setuptools import find_packages

DISTNAME = ''
DESCRIPTION = "Random Forest microscope"
URL = 'https://github.com/shifwang/RFmicroscope'
MAINTAINER = 'Yu Wang'
MAINTAINER_EMAIL = 'wang.yu@berkeley.edu'
LICENSE = 'new BSD'
VERSION = '0.1.3'


def configuration(parent_package='rfms', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    #config.add_subpackage('rfms')

    return config

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    print(local_path, old_path)
    os.chdir(local_path)
    sys.path.insert(0, local_path)
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          packages=find_packages(),
          include_package_data=True,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
            ],
          install_requires=["numpy", "pandas", "scipy", "scikit-learn>=0.18", "cython"],
          setup_requires=["cython"])
