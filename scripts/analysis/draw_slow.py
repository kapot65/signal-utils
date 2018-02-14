#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:32:51 2018

@author: chernov
"""
import csv
import io
import re
from argparse import ArgumentParser, ArgumentTypeError
from glob import glob
from os import path

import dateutil
import dfparser  # Numass files parser
import matplotlib.pyplot as plt  # plotting library
import seaborn as sns  # matplotlib grahs visual enchancer
from dateutil.parser import parse as dateparse  # timestamp parser


def parse_args():
    """Parse arguments from command line."""
    def mark(val):
        """Mark parser type.

        Type represents time mark with label in format: LABEL,TIME_ISO
        """
        try:
            label, timestamp = val.split(',')
            return label, dateparse(timestamp)
        except ValueError:
            raise ArgumentTypeError("Coordinates must be LABEL,TIMESTAMP_ISO")

    parser = ArgumentParser(description='Slow control graph drawer')
    parser.add_argument('input', help='Input dataforge file.')
    parser.add_argument('-t', '--title', type=str, default=None,
                        help='Graph title (default - filename).'
                        'The flag can be used repeatedly in command.')
    parser.add_argument('--x-scale', default=None,
                        choices=['linear', 'log', 'logit', 'symlog'],
                        help='Axis x scale (default - linear).')
    parser.add_argument('--y-scale', default=None,
                        choices=['linear', 'log', 'logit', 'symlog'],
                        help='Axis y scale (default - linear).')
    parser.add_argument('-e', '--exclude', action='append', default=[],
                        help='Exclude graphs names. '
                        'The flag can be used repeatedly in command.')
    parser.add_argument('-m', '--mark', action='append', type=mark, default=[],
                        help='Add mark with label on graph in '
                        'LABEL,TIMESTAMP_ISO format '
                        '(ex: -m mark1,2017-11-19T09:44:11Z). '
                        'The flag can be used repeatedly in command.')
    return parser.parse_args()


def _main():
    args = parse_args()
    # Read dataforge point

    # Creating graph
    _, axes = plt.subplots()

    for idx_f, input_file in enumerate(glob(args.input)):
        header, _, data = dfparser.parse_from_file(input_file)
        # Read binary data manually due to machine header error
        # Dont need for good data
        header_len = dfparser.type_codes.ENVELOPE_HEADER_CODES[
            header["type"]]["header_len"]

        with (open(input_file, 'rb')) as raw_file:
            raw_file.seek(header_len + header['meta_len'])
            data = raw_file.read()

        # Prettify data. Remove double spaces
        data_tabed = re.sub(b'[ \t]+', b'\t', data)
        # Remove icorrect column name '#f\t' from data
        data_tabed = data_tabed[3:]

        # Parse data as TSV format
        tsv = csv.DictReader(io.StringIO(
            data_tabed.decode()), delimiter='\t')

        # Extracting values from parsed TSV
        graphs = {}
        for row in tsv:
            for key in row.keys():
                if key != 'timestamp':
                    if row[key] != '@null':  # Filter null values
                        if key not in graphs:
                            graphs[key] = {'x': [], 'y': []}

                        # Append timestamp to point
                        graphs[key]['x'].append(
                            dateutil.parser.parse(row['timestamp']))
                        # Append value to point
                        graphs[key]['y'].append(
                            float(row[key]))

        # Plotting graphs
        for idx, graph in enumerate(graphs.keys()):
            if graph not in args.exclude:  # Filter excluded data
                label = None
                if idx_f == 1:
                    label = graph
                axes.plot(
                    graphs[graph]['x'], graphs[graph]['y'],
                    color=sns.color_palette()[idx], label=label)

    for text, x_coord in args.mark:
        axes.annotate(text, xy=(x_coord, 0), xytext=(x_coord, 1e2),
                      arrowprops=dict(facecolor='black', shrink=0.05))

    # Applying parameters
    if args.title:
        axes.set_title(args.title)
    else:
        axes.set_title(path.basename(args.input))
    if args.x_scale:
        axes.set_xscale(args.x_scale)
    if args.y_scale:
        axes.set_yscale(args.y_scale)
    axes.legend()
    plt.show()


if __name__ == "__main__":
    sns.set_context("poster")
    _main()
