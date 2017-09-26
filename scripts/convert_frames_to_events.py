"""Dataforge point frames to events converting tool."""

from argparse import ArgumentParser
from os import path

import dfparser

from signal_utils.convert_utils import df_frames_to_events
from signal_utils.extract_utils import extract_amps_approx2


def _parse_args():
    parser = ArgumentParser(description='Dataforge point frames to '
                            'events converting tool.')

    parser.add_argument('input', help='Input dataforge file.')
    parser.add_argument('-o', '--output', default=None,
                        help='Output dataforge file. '
                        '(default - %filename%_extr.df')
    parser.add_argument('--frame-l', default=15,
                        help='Doubles left frame size. (default - 15)')
    parser.add_argument('--frame-r', default=25,
                        help='Doubles right frame size. (default - 25)')

    return parser.parse_args()


def _main():
    args = _parse_args()

    if args.output is None:
        output = path.join(path.dirname(args.input),
                           "%s_extr.df" % (path.splitext(args.input)[0]))
    else:
        output = args.output

    _, meta, data = dfparser.parse_from_file(args.input)
    meta_out, data_out = df_frames_to_events(meta, data, extract_amps_approx2,
                                             frame_l=args.frame_l,
                                             frame_r=args.frame_r)
    with open(output, "wb") as out_file:
        out_file.write(dfparser.create_message(meta_out, data_out))


if __name__ == "__main__":
    _main()
