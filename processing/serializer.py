"""Intermediate results serialization methods."""
import msgpack
import msgpack_numpy as m

m.patch()


def pack(s):
    """Msgpack-numpy serialize."""
    return msgpack.packb(s, use_bin_type=True)


def unpack(s):
    """Msgpack-numpy parse."""
    return msgpack.unpackb(s, encoding='utf-8')
