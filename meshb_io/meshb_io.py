import os
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_char_p, c_int, c_int64, POINTER

lib = npct.load_library("_meshb_io.so", os.path.dirname(__file__))

npflags = ["C_CONTIGUOUS"]
coords_type = npct.ndpointer(dtype=np.float64, ndim=2, flags=npflags)
conn_type = npct.ndpointer(dtype=np.int64, ndim=2, flags=npflags)
tags_type = npct.ndpointer(dtype=np.intc, ndim=1, flags=npflags)

lib.open_file_write.argtypes = [c_char_p, c_int]
lib.open_file_write.restype = c_int64

lib.open_file_read.argtypes = [c_char_p, POINTER(c_int), POINTER(c_int)]
lib.open_file_read.restype = c_int64

lib.close_file.argtypes = [c_int64]
lib.close_file.restype = None

lib.write_vertices.argtypes = [c_int64, c_int64, c_int, coords_type]
lib.write_vertices.restype = c_int

lib.write_elements.argtypes = [
    c_int64,
    c_int64,
    c_int,
    c_int,
    conn_type,
    tags_type,
]
lib.write_elements.restype = c_int

lib.get_num_entities.argtypes = [c_int64]
lib.get_num_entities.restype = c_int64

lib.read_vertices.argtypes = [c_int64, c_int64, c_int, coords_type]
lib.read_vertices.restype = c_int

lib.read_elements.argtypes = [
    c_int64,
    c_int64,
    c_int,
    c_int,
    conn_type,
    tags_type,
]
lib.read_elements.restype = c_int

VERTEX = 4
EDGE = 5
EDGE2 = 25
TRIANGLE = 6
TRIANGLE2 = 24
TETRAHEDRON = 8
TETRAHEDRON2 = 30

ELEM_NUM_VERTS = {
    EDGE: 2,
    EDGE2: 3,
    TRIANGLE: 3,
    TRIANGLE2: 6,
    TETRAHEDRON: 4,
    TETRAHEDRON2: 10,
}


class MeshbWriter:

    def __init__(self, fname, dim):

        self._file = lib.open_file_write(fname.encode(), dim)
        self._dim = dim

    def __del__(self):

        lib.close_file(self._file)

    def write_vertices(self, verts):

        assert verts.shape[1] == self._dim
        res = lib.write_vertices(
            self._file,
            verts.shape[0],
            verts.shape[1],
            np.ascontiguousarray(verts, dtype=np.float64),
        )
        assert res != -1

    def write_elements(self, etype, conn, tags):

        res = lib.write_elements(
            self._file,
            conn.shape[0],
            conn.shape[1],
            etype,
            np.ascontiguousarray(conn, dtype=np.int64),
            np.ascontiguousarray(tags, dtype=np.intc),
        )
        assert res != -1


class MeshbReader:

    def __init__(self, fname):

        dim = c_int(-1)
        ver = c_int(-1)
        self._file = lib.open_file_read(fname.encode(), ver, dim)
        self._dim = dim.value
        self._ver = ver.value
        assert self._ver == 3

    def __del__(self):

        lib.close_file(self._file)

    def _num_vertices(self):

        return lib.get_num_entities(self._file, VERTEX)

    def read_vertices(self):

        n = self._num_vertices()
        verts = np.zeros((n, self._dim), dtype=np.float64)
        lib.read_vertices(self._file, n, self._dim, verts)
        return verts

    def _num_elements(self, etype):

        return lib.get_num_entities(self._file, etype)

    def _read_elements(self, etype):

        n = self._num_elements(etype)
        m = ELEM_NUM_VERTS[etype]
        conn = np.zeros((n, m), dtype=np.int64)
        tags = np.zeros(n, dtype=np.intc)
        lib.read_elements(self._file, n, m, etype, conn, tags)
        return conn, tags

    def read_elements(self):

        return {
            x: self._read_elements(x)
            for x in ELEM_NUM_VERTS.keys()
            if self._num_elements(x) > 0
        }
