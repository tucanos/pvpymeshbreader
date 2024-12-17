import os
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_char_p, c_int, c_int64, POINTER

lib = npct.load_library("_meshb_io.so", os.path.dirname(__file__))

npflags = ["C_CONTIGUOUS"]
coords_type = npct.ndpointer(dtype=np.float64, ndim=2, flags=npflags)
conn_type = npct.ndpointer(dtype=np.int64, ndim=2, flags=npflags)
conn_type_int = npct.ndpointer(dtype=np.intc, ndim=2, flags=npflags)
tags_type = npct.ndpointer(dtype=np.intc, ndim=1, flags=npflags)

lib.open_file_write.argtypes = [c_char_p, c_int]
lib.open_file_write.restype = c_int64

lib.open_file_read.argtypes = [c_char_p, POINTER(c_int), POINTER(c_int)]
lib.open_file_read.restype = c_int64

lib.close_file.argtypes = [c_int64]
lib.close_file.restype = None

lib.write_vertices.argtypes = [c_int64, c_int64, c_int, coords_type]
lib.write_vertices.restype = c_int

lib.write_sol.argtypes = [c_int64, c_int, c_int, c_int64, c_int, coords_type]
lib.write_sol.restype = c_int

lib.write_elements.argtypes = [
    c_int64,
    c_int64,
    c_int,
    c_int,
    conn_type,
    tags_type,
]
lib.write_elements.restype = c_int

lib.get_num_entities.argtypes = [c_int64, c_int]
lib.get_num_entities.restype = c_int64

lib.get_sol_info.argtypes = [c_int64, c_int, POINTER(c_int)]
lib.get_sol_info.restype = c_int64

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

lib.read_elements_int.argtypes = [
    c_int64,
    c_int64,
    c_int,
    c_int,
    conn_type_int,
    tags_type,
]
lib.read_elements_int.restype = c_int

lib.read_sol.argtypes = [
    c_int64,
    c_int,
    c_int64,
    c_int,
    coords_type,
]
lib.read_sol.restype = c_int

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

SOL_AT_VERTICES = 62
SOL_AT_EDGES = 63
SOL_AT_TRIANGLES = 64
SOL_AT_TETRAHEDRA = 74


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

    def write_solution(self, loc, sol):

        if loc == VERTEX:
            loc = SOL_AT_VERTICES
        elif loc == EDGE or loc == EDGE2:
            loc = SOL_AT_EDGES
        elif loc == TRIANGLE or loc == TRIANGLE2:
            loc = SOL_AT_TRIANGLES
        elif loc == TETRAHEDRON or loc == TETRAHEDRON2:
            loc = SOL_AT_TETRAHEDRA
        else:
            raise NotImplementedError()

        lib.write_sol(
            self._file,
            self._dim,
            loc,
            sol.shape[0],
            sol.shape[1],
            np.ascontiguousarray(sol, dtype=np.float64),
        )


class MeshbReader:

    def __init__(self, fname):

        if not os.path.exists(fname):
            raise IOError(f"File {fname} does not exist")

        dim = c_int(-1)
        ver = c_int(-1)
        self._file = lib.open_file_read(fname.encode(), ver, dim)
        if self._file == 0:
            raise IOError(f"Unable to open {fname} ")
        self._dim = dim.value
        self._ver = ver.value
        assert self._ver in [2, 3, 4]
        assert self._dim in [2, 3]

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
        tags = np.zeros(n, dtype=np.intc)
        if self._ver <= 3:
            conn = np.zeros((n, m), dtype=np.intc)
            lib.read_elements_int(self._file, n, m, etype, conn, tags)
        else:
            conn = np.zeros((n, m), dtype=np.int64)
            lib.read_elements(self._file, n, m, etype, conn, tags)
        return conn, tags

    def read_elements(self):

        return {
            x: self._read_elements(x)
            for x in ELEM_NUM_VERTS.keys()
            if self._num_elements(x) > 0
        }

    def read_solution(self, loc):

        if loc == VERTEX:
            loc = SOL_AT_VERTICES
        elif loc == EDGE or loc == EDGE2:
            loc = SOL_AT_EDGES
        elif loc == TRIANGLE or loc == TRIANGLE2:
            loc = SOL_AT_TRIANGLES
        elif loc == TETRAHEDRON or loc == TETRAHEDRON2:
            loc = SOL_AT_TETRAHEDRA
        else:
            raise NotImplementedError()

        m = c_int(-1)
        n = lib.get_sol_info(self._file, loc, m)
        m = m.value

        sol = np.zeros((n, m), dtype=np.float64)
        lib.read_sol(self._file, loc, n, m, sol)
        return sol
