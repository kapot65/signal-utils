"""Signal utils init script."""
import sys
from os import path

from pkg_resources import get_distribution

import convert_utils
import draw_utils
import extract_utils
import gdrive_utils
import generation_utils
import test_utils

MAIN_DIR = path.abspath(path.dirname(__file__))
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)
del MAIN_DIR


__version__ = get_distribution('dfparser').version
