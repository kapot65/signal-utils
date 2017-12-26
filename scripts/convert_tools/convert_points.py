# -*- coding: utf-8 -*-
"""Скрипт для конвертирования фреймов в события в df файлах."""
# pylint: disable-msg=E0611

from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os import path
import sys
import struct

from contextlib import closing
import dfparser
from multiprocess import Pool
from tqdm import tqdm

MAIN_DIR = path.abspath(path.join(path.dirname(__file__), '..'))
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)
del MAIN_DIR

from signal_utils.convert_utils import df_frames_to_events
from signal_utils.extract_utils import extract_amps_approx


def parse_args():
    """Парсинг параметров."""
    parser = ArgumentParser(description='Dataforge Lan10-12PCI frames to '
                            'events converter.')

    parser.add_argument('data_path', help='Path to root folder with data.')
    parser.add_argument('out_path', help='Path to root output folder.')
    parser.add_argument('wildcard',
                        help='Files wildcard relative to root folder.')

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()

    FILES_ABS_WILDCARD = path.join(ARGS.data_path, ARGS.wildcard)
    FILES = glob(FILES_ABS_WILDCARD, recursive=True)

    def process_file(data_file):
        """Обработка отдельного файла."""
        try:
            filepath_rel = path.relpath(data_file, ARGS.data_path)
            filepath_out = path.join(ARGS.out_path, filepath_rel)
            _, meta_real, data_real = dfparser.parse_from_file(data_file)
            meta_real_, data_real_ = df_frames_to_events(meta_real, data_real,
                                                         extract_amps_approx,
                                                         correct_time=True)

            if not path.exists(path.dirname(filepath_out)):
                makedirs(path.dirname(filepath_out))

            with open(filepath_out, "wb") as out_file:
                out_file.write(dfparser.create_message(meta_real_, data_real_))
        except struct.error:
            pass


    with closing(Pool()) as pool:
        with tqdm(total=len(FILES)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(process_file,
                                                           FILES))):
                pbar.update()
