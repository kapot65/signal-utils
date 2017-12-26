# TODO: remove hardcode
from glob import glob
from os import makedirs
from os import path
import shutil

lan10_root = "/data/lan10"
lan10_processed_root = "/data/lan10_processed/"

metas = [
    path.relpath(p, lan10_root) for p in
    glob(path.join(lan10_root, "**", "meta"), recursive=True)]
    
scenarios = [
    path.relpath(p, lan10_root) for p in
    glob(path.join(lan10_root, "**", "scenario"), recursive=True)]
    
voltages = [
    path.relpath(p, lan10_root) for p in
    glob(path.join(lan10_root, "**", "voltage"), recursive=True)]
    
for metafile in [*metas, *scenarios, *voltages]:
    src = path.join(lan10_root, metafile)
    dst = path.dirname(path.join(lan10_processed_root, metafile))
    if not path.exists(dst):
        makedirs(dst)
    shutil.copy(src, dst)
