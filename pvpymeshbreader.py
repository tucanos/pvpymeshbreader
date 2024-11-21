import logging
from paraview.util.vtkAlgorithm import (
    smproxy,
    VTKPythonAlgorithmBase,
    smproperty,
    smdomain,
    smhint,
)
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.util.vtkConstants import (
    VTK_ID_TYPE,
    VTK_UNSIGNED_CHAR,
    VTK_LINE,
    VTK_TRIANGLE,
    VTK_TETRA,
    VTK_POLYGON,
    VTK_QUADRATIC_TRIANGLE,
    VTK_QUADRATIC_EDGE,
    VTK_QUADRATIC_TETRA,
)
from vtkmodules.vtkCommonCore import vtkDataArraySelection
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet
from vtk.util.numpy_support import numpy_to_vtk
import logging
import json
import os, sys

sys.path.append(os.path.dirname(__file__))
from meshb_io import (
    MeshbReader,
    EDGE,
    EDGE2,
    TRIANGLE,
    TRIANGLE2,
    TETRAHEDRON,
    TETRAHEDRON2,
)


# Map meshb_io type to (VTK type, element dimension)
CELL_TYPES = {
    EDGE: (VTK_LINE, 1),
    EDGE2: (VTK_QUADRATIC_EDGE, 1),
    TRIANGLE: (VTK_TRIANGLE, 2),
    TRIANGLE2: (VTK_QUADRATIC_TRIANGLE, 2),
    TETRAHEDRON: (VTK_TETRA, 3),
    TETRAHEDRON2: (VTK_QUADRATIC_TETRA, 3),
}


def _numpy_to_cell_array(cell_types, offset, connectivity):
    """
    Create a vtkCellArray from 2 numpy arrays and a vtkUnsignedCharArray cell type
    array from a numpy array
    """
    ca = vtkCellArray()
    ca.SetData(
        numpy_to_vtk(offset, deep=1, array_type=VTK_ID_TYPE),
        numpy_to_vtk(connectivity, deep=1, array_type=VTK_ID_TYPE),
    )
    ct = numpy_to_vtk(cell_types, deep=1, array_type=VTK_UNSIGNED_CHAR)
    return ct, ca


@smproxy.reader(
    name="PythonMeshbReader",
    label="Python-based meshb Reader",
    extensions=["mesh", "meshb", "json"],
    file_description="Meshb files",
)
class PythonCODAReader(VTKPythonAlgorithmBase):
    """A reader that reads meshb files"""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet"
        )
        self._filename = None

        def createModifiedCallback(anobject):
            import weakref

            weakref_obj = weakref.ref(anobject)
            anobject = None

            def _markmodified(*args, **kwars):
                o = weakref_obj()
                if o is not None:
                    o.Modified()

            return _markmodified

        self._arrayselection = vtkDataArraySelection()
        self._arrayselection.AddObserver("ModifiedEvent", createModifiedCallback(self))

        self._names = None
        self._mesh = None
        self._sols = {}

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions=["mesh", "meshb", "json"], file_description="files")
    def SetFileName(self, name):
        """Specify filename for the file to read."""
        if self._filename != name:
            self._filename = name
            logging.info(f"Reading {name}")
            if name.endswith(".mesh") or name.endswith(".meshb"):
                self._mesh = MeshbReader(name)
            elif name.endswith(".json"):
                with open(name, "r") as f:
                    config = json.load(f)
                mesh_name = config["mesh"]
                print(mesh_name)
                logging.info(f"Mesh file: {mesh_name}")
                self._mesh = MeshbReader(mesh_name)
                if "names" in config:
                    self._names = {tag: name for name, tag in config["names"].items()}
                    logging.info(f"Names: {self._names}")
                if "fields" in config:
                    for var_name, sol_fname in config["fields"].items():
                        logging.info(f"Field {var_name}: {sol_fname}")
                        self._sols[var_name] = MeshbReader(sol_fname)
            else:
                pass
            self.Modified()

    def _get_timesteps(self):
        return None

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self):
        return self._get_timesteps()

    # Array selection API is typical with readers in VTK
    # This is intended to allow ability for users to choose which arrays to
    # load. To expose that in ParaView, simply use the
    # smproperty.dataarrayselection().
    # This method **must** return a `vtkDataArraySelection` instance.
    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self):

        return self._arrayselection

    def RequestInformation(self, request, inInfoVec, outInfoVec):

        for var_name in self._sols.keys():
            self._arrayselection.AddArray(var_name)
            if self._first_load:
                self._arrayselection.EnableArray(var_name)
        self._first_load = False
        return 1

    def EnableAllVariables(self):

        self._arrayselection.EnableAllArrays()
        self._first_load = False
        self.Modified()

    def _create_unstructured_grid(self, dim, tag, xyz, cells):

        logging.debug("create a vtkUnstructuredGrid")
        flg = np.zeros(xyz.shape[0], dtype=bool)
        for etype, (_, edim) in CELL_TYPES.items():
            if edim == dim and etype in cells:
                conn, etags = cells[etype]
                (iels,) = np.nonzero(etags == tag)
                conn = conn[iels, :]
                ids = conn.ravel()
                flg[ids] = True
        (used_verts,) = np.nonzero(flg)
        if used_verts.size == 0:
            return None, None

        new_idx = np.zeros(xyz.shape[0], dtype=np.int64) - 1
        new_idx[used_verts] = np.arange(used_verts.size)

        ug = vtkUnstructuredGrid()
        pug = dsa.WrapDataObject(ug)
        pug.SetPoints(xyz[used_verts, :])

        cell_types = np.zeros(0, dtype=np.int64)
        offsets = np.array([0], dtype=np.int64)
        connectivity = np.zeros(0, dtype=np.int64)
        tags = np.zeros(0, dtype=np.int64)
        for etype, (vtk_type, edim) in CELL_TYPES.items():
            if edim == dim and etype in cells:
                conn, etags = cells[etype]
                flg = etags == tag
                n = np.sum(flg)
                start = offsets[-1]
                if n > 0:
                    logging.info("adding %d elements %d" % (n, etype))
                    conn = conn[flg, :]
                    _, m = conn.shape
                    ptr = start + m * (1 + np.arange(n))
                    offsets = np.append(offsets, ptr)
                    connectivity = np.append(connectivity, new_idx[conn.ravel()])
                    cell_types = np.append(
                        cell_types, vtk_type * np.ones(n, dtype=np.int64)
                    )
                    tags = np.append(tags, tag * np.ones(n))

        ca = vtkCellArray()
        ca.SetNumberOfCells(offsets.size - 1)
        ca.SetData(
            numpy_to_vtk(offsets, deep=1, array_type=VTK_ID_TYPE),
            numpy_to_vtk(connectivity, deep=1, array_type=VTK_ID_TYPE),
        )
        ct = numpy_to_vtk(cell_types, deep=1, array_type=VTK_UNSIGNED_CHAR)
        ug.SetCells(ct, ca)

        data = pug.GetCellData()
        data.append(tags, "tag")

        return pug

    def RequestData(self, request, inInfoVec, outInfoVec):

        xyz = self._mesh.read_vertices()
        cells = self._mesh.read_elements()
        cell_dim = max([CELL_TYPES[etype][1] for etype in cells.keys()])

        mbds = vtkMultiBlockDataSet.GetData(outInfoVec, 0)
        iz = 0
        for dim in [cell_dim, cell_dim - 1]:
            tags = [
                np.unique(tags)
                for etype, (_, tags) in cells.items()
                if CELL_TYPES[etype][1] == dim
            ]
            if len(tags) == 0:
                continue
            tags = np.unique(np.concatenate(tags))
            for tag in tags:
                try:
                    name = self._names[tag]
                except:
                    name = "tag_%d" % tag
                pug = self._create_unstructured_grid(dim, tag, xyz, cells)
                if pug is not None:
                    logging.info(
                        "%s: %d vertices, %d cells"
                        % (name, pug.GetNumberOfPoints(), pug.GetNumberOfCells())
                    )
                    mbds.SetBlock(iz, pug.VTKObject)
                    mbds.GetMetaData(iz).Set(mbds.NAME(), name)
                    iz += 1
                    # data = pug.GetCellData()
                    # added = []
                    # if dim == cell_dim:
                    #     for name, arr in zip(*cell_data):
                    #         if name not in added:
                    #             data.append(arr[cell_ids], name)
                    # elif dim == cell_dim - 1:
                    #     for name, arr in zip(*bdy_data):
                    #         if name not in added:
                    #             data.append(arr[cell_ids], name)

        return 1


def test(fname):
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter

    reader = PythonCODAReader()
    reader.SetFileName(fname)
    reader.EnableAllVariables()
    reader.Update()

    grid = reader.GetOutputDataObject(0)
    nb = grid.GetNumberOfBlocks()
    if nb == 0:
        logging.error("No block created")
        quit()

    for i in range(nb):
        blk = grid.GetBlock(i)

        logging.info(
            "Block %d: %d points, %d cells"
            % (i, blk.GetNumberOfPoints(), blk.GetNumberOfCells())
        )
        if blk.GetNumberOfCells() == 0:
            logging.error("No cell in block %d" % i)
            quit()
        if blk.GetNumberOfPoints() == 0:
            logging.error("No point in block %d" % i)
            quit()

        data = blk.GetCellData()
        na = data.GetNumberOfArrays()
        for i in range(na):
            logging.info("Cell data %s" % data.GetArrayName(i))

    fname = fname.replace(".meshb", ".vtm").replace(".json", ".vtm")
    logging.info("Writing %s" % fname)
    writer = vtkXMLMultiBlockDataWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetDataModeToAscii()
    writer.SetCompressorTypeToNone()
    # writer.SetDataModeToAppended()
    # writer.EncodeAppendedDataOff()
    writer.Update()


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.DEBUG)

    # test("quadratic.meshb")
    test("test.json")
