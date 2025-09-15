from __future__ import print_function

import os
import sys


from setuptools import setup, find_packages



HERE = os.path.dirname(os.path.abspath(__file__))

# import ``__version__` from code base
exec(open(os.path.join(HERE, 'dynrdpg', 'version.py')).read())


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)




DISTNAME = 'Dynamic RDPG'
DESCRIPTION = 'Dynamic RDPG'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Joshua D. Loyal'
MAINTAINER_EMAIL = 'jloyal25@gmail.com'
URL = 'https://joshloyal.github.io/dynrdpg'
DOWNLOAD_URL = 'https://pypi.org/project/dynrdpg/#files'
LICENSE = 'MIT'
VERSION = __version__
CLASSIFIERS = []


def setup_package():
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
    )
if __name__ == '__main__':
    setup_package()
