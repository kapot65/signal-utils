"""Correct set numbers between MADC and Lan10 datasets.

Script will:
- Correct set numbers and adds metadata to Lan10 sets.
  Detached from MADC sets will be marked.
- Compress points binary data.

"""
from functools import partial
from contextlib import closing
from multiprocessing import Pool
import glob
import shutil
import struct
import zlib
from argparse import ArgumentParser
from enum import Enum
from os import listdir, makedirs, path

import dfparser
from dateutil import parser as timeparser
from natsort import natsorted


def __parse_args():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument('LAN10_PATH', help='Path to Lan10 dataset.')
    parser.add_argument('MADC_PATH', help='Path to MADC dataset.')
    parser.add_argument('OUT_PATH', help='Output data root path.')
    parser.add_argument('--err-sec', '-e', type=float, default=10.0,
                        help='Error in seconds between datasets.')
    parser.add_argument('-t', '--threads', type=int, default=None,
                        help='Processing threads (default - cpu cores count).')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite files in output directory.')

    return parser.parse_args()


def read_madc_sets(madc_sets_raw, group_abs_path):
    """Read begin and end times for each set in group.

    Parameters
    ----------
    madc_sets_raw : list
       Sets list for current group.
    group_abs_path : str
       Group absolute path.
    Returns
    -------
    madc_sets : dictionary
       A dictionary keyed by sets containing begin and end times.

    """
    madc_sets = {}
    for madc_set in madc_sets_raw:
        files = listdir(
            path.join(group_abs_path, madc_set))
        points = natsorted([f for f in files if f.startswith('p')])

        if points:
            _, meta_p_0, _ = dfparser.parse_from_file(
                path.join(group_abs_path, madc_set,
                          points[0]), nodata=True)

            _, meta_p_last, _ = dfparser.parse_from_file(
                path.join(group_abs_path, madc_set,
                          points[-1]), nodata=True)

            madc_sets[madc_set] = {
                "begin": timeparser.parse(meta_p_0["start_time"][0]),
                "end": timeparser.parse(meta_p_last["end_time"][-1])
            }

    return madc_sets


class SetState(Enum):
    """Set match state."""

    CORRUPTED = 1
    EMPTY = 2
    DETACHED = 3


def match_lan10_sets(lan10_sets_raw, lan10_root,
                     lan10_group_rel_path, madc_sets, err_sec):
    """Find correspondense between lan10 and madc sets.

    Parameters
    ----------
    lan10_sets_raw : list
       Lan10 sets list for current group.
    lan10_group_abs_path : str
       Lan10 group absolute path.
    madc_sets : str
       MADC sets time borders dictionary (return value from read_madc_sets).
    err_sec : float
       Maximal error between borders in seconds.
    Returns
    -------
    corrs : dictionary
       Sets mapping.

    """
    corrs = []
    for lan10_set in lan10_sets_raw:
        set_abs_path = path.join(lan10_root, lan10_group_rel_path, lan10_set)
        set_rel_path = path.relpath(set_abs_path, lan10_root)
        files = listdir(set_abs_path)
        points = natsorted([f for f in files if f.startswith('p')])
        if points:
            check_beginning = True
            try:
                _, meta, _ = dfparser.parse_from_file(
                    path.join(set_abs_path, points[0]), nodata=True)
                begin = timeparser.parse(meta["params"]["start_time"])
            except struct.error:
                check_beginning = False

            check_ending = True
            try:
                _, meta, _ = dfparser.parse_from_file(
                    path.join(set_abs_path, points[-1]), nodata=True)
                end = timeparser.parse(meta["params"]["end_time"])
            except struct.error:
                check_ending = False

            if not (check_beginning or check_ending):

                corrs.append({"from": set_rel_path, "to": SetState.CORRUPTED})
            else:
                detached = True
                for key in madc_sets:
                    if check_beginning:
                        begin_err = abs(
                            (madc_sets[key]["begin"] - begin)
                            .total_seconds()
                        )
                    if check_ending:
                        end_err = abs(
                            (madc_sets[key]["end"] -
                             end).total_seconds()
                        )
                    if (not check_beginning or begin_err < err_sec) and \
                            (not check_ending or end_err < err_sec):

                        corrs.append({
                            "from": set_rel_path,
                            "to": path.join(path.dirname(set_rel_path), key)
                        })
                        detached = False
                        break

                if detached:
                    corrs.append({
                        "from": set_rel_path,
                        "to": SetState.DETACHED
                    })
        else:
            corrs.append({
                "from": set_rel_path,
                "to": SetState.EMPTY
            })
    return corrs


def process_set(lan10_root, madc_root, out_root, overwrite, corrs):
    """Process group dataset.

    Parameters
    ----------
    lan10_root : str
       Lan10 data root path.
    out_root : str
       Output data root path.
    corrs : list
       Correspondency list between sets (return value from match_lan10_sets).

    """
    for corr in corrs:
        if corr["to"] == SetState.CORRUPTED:
            out_path = path.join(out_root, "%s-corrupted" % corr["from"])
            if path.exists(out_path):
                if overwrite:
                    shutil.rmtree(out_path)
                else:
                    continue
            shutil.copytree(path.join(lan10_root, corr["from"]), out_path)

        elif corr["to"] == SetState.DETACHED:
            out_path = path.join(out_root, "%s-detached" % corr["from"])
            if path.exists(out_path):
                if overwrite:
                    shutil.rmtree(out_path)
                else:
                    continue
            shutil.copytree(path.join(lan10_root, corr["from"]), out_path)

        elif corr["to"] == SetState.EMPTY:
            out_path = path.join(out_root, "%s-empty" % corr["from"])
            if path.exists(out_path):
                if overwrite:
                    shutil.rmtree(out_path)
                else:
                    continue
            shutil.copytree(path.join(lan10_root, corr["from"]), out_path)

        else:
            lan10_set_path = path.join(lan10_root, corr["from"])
            madc_set_path = path.join(madc_root, corr["to"])
            out_set_path = path.join(out_root, corr["to"])
            if path.exists(out_set_path):
                if overwrite:
                    shutil.rmtree(out_set_path)
                else:
                    continue

            makedirs(out_set_path)

            for metafile in ["meta", "voltage", "scenario"]:
                if path.exists(path.join(madc_set_path, metafile)):
                    shutil.copy(path.join(madc_set_path, metafile),
                                path.join(out_set_path, metafile))

            for point in listdir(lan10_set_path):
                try:
                    _, meta, data = dfparser.parse_from_file(
                        path.join(lan10_set_path, point)
                    )
                    meta["compression"] = "zlib"
                    with open(path.join(out_set_path, point), "wb") as out_file:
                        out_file.write(
                            dfparser.create_message(
                                meta, zlib.compress(data)
                            )
                        )
                except struct.error:
                    shutil.copy(
                        path.join(lan10_set_path, point),
                        path.join(out_set_path, "%s-corrupted" % point)
                    )


def process_group(group, args):
    """Process all sets in group."""
    madc_sets_raw = [
        f for f in listdir(path.join(args.MADC_PATH, group))
        if f.startswith("set_")]
    lan10_sets_raw = [
        f for f in listdir(path.join(args.LAN10_PATH, group))
        if f.startswith("set_")]

    madc_sets = read_madc_sets(
        madc_sets_raw, path.join(args.MADC_PATH, group)
    )

    corrs = match_lan10_sets(
        lan10_sets_raw, args.LAN10_PATH, group, madc_sets, args.err_sec
    )

    process_set(args.LAN10_PATH, args.MADC_PATH, args.OUT_PATH,
                args.overwrite, corrs)


def __main():
    args = __parse_args()

    lan10_groups = [path.relpath(g, args.LAN10_PATH) for g in list(
        set([path.dirname(s) for s in glob.glob(
            path.join(args.LAN10_PATH, "**", "set_*"), recursive=True)])
    )]

    madc_groups = [path.relpath(g, args.MADC_PATH) for g in list(
        set([path.dirname(s) for s in glob.glob(
            path.join(args.MADC_PATH, "**", "set_*"), recursive=True)])
    )]

    merging_groups = [
        grp for grp in lan10_groups if grp in madc_groups]

    process_group_part = partial(process_group, args=args)
    with closing(Pool(args.threads)) as pool:
        pool.map(process_group_part, merging_groups)


if __name__ == "__main__":
    __main()
