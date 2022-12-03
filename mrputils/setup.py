import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'mrputils'
AUTHOR = 'Amir Shamsa'
AUTHOR_EMAIL = 'amirshamsa@gmail.com'
URL = 'https://github.com/scienclick/pde_cap_mrp_zagros/tree/main/mrputils'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'This is a util module to help with movie revenue prediction'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'scikit-learn',
      'nltk',

]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
