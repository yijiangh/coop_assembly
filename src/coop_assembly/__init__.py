"""
*************
coop_assembly
*************

.. currentmodule:: coop_assembly

.. toctree::
    :maxdepth: 2

    coop_assembly.data_structure
    coop_assembly.geometry_generation
    coop_assembly.help_functions
    coop_assembly.planning

"""
    # coop_assembly.assembly_info_generation
    # coop_assembly.choreo_interface

from __future__ import print_function

from .__version__ import __author__, __author_email__, __copyright__, \
                        __description__, __license__, __title__, __url__, __version__

import os, logging

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, 'data'))


def _find_resource(filename):
    filename = filename.strip('/')
    return os.path.abspath(os.path.join(DATA, filename))


def get_data(filename):
    return _find_resource(filename)

__all__ = [
    '__author__', '__author_email__', '__copyright__', '__description__',
    '__license__', '__title__', '__url__', '__version__', 'get_data',
]

def get_logger(name):
    logger = logging.getLogger(name)

    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter("%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
                                     datefmt=None,
                                     reset=True,
                                     log_colors={'DEBUG': 'cyan', 'INFO': 'green',
                                                 'WARNING': 'yellow',
                                                 'ERROR': 'red', 'CRITICAL': 'red',
                                                 }
                                     )
    except ImportError:
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

LOGGER = get_logger('coop_assembly')
