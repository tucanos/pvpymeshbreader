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
    VERTEX,
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
    extensions=["mesh", "meshb"],
    file_description="Meshb files",
)
class PythonMeshbReader(VTKPythonAlgorithmBase):
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

        self._file_names = []
        self._filename = None
        self._current_time = None
        self._timesteps = None

        self._names = None
        self._mesh = None
        self._node_sols = {}
        self._cell_sols = {}
        self._first_load = True

    @smproperty.stringvector(
        name="FileNames",
        label="File Names",
        animateable="1",
        # clean_command="RemoveAllFileNames",
        command="AddFileName",
        repeat_command="1",
        number_of_elements="1",
        panel_visibility="never",
    )
    @smdomain.filelist()
    @smhint.filechooser(extensions=["mesh", "meshb"], file_description="files")
    def AddFileName(self, name):

        if name != "None":
            self._file_names.append(name)
            self.__read_timesteps(name)

    def __read_timesteps(self, name):

        prefix = name.replace(".meshb", "").replace(".mesh", "")
        config_fname = prefix + ".json"
        if os.path.exists(config_fname):
            logging.info(f"Config file: {config_fname}")
            with open(config_fname, "r") as f:
                config = json.load(f)
            if "steps" in config:
                assert self._timesteps is None
                self._timesteps = [x["time"] for x in config["steps"]]

    def __read_files_for_step(self, time_step):

        if len(self._file_names) > 1:
            if time_step is None:
                time_step = 0
            name = self._file_names[time_step]
        else:
            name = self._file_names[0]
            if time_step is not None:
                time_step = self._timesteps.index(time_step)
            else:
                time_step = 0

        if self._mesh is None or self._filename is None or self._filename != name:
            self._filename = name
            logging.info(f"Reading {name}")
            if name.endswith(".mesh") or name.endswith(".meshb"):
                logging.info(f"Mesh file: {name}")
                self._mesh = MeshbReader(name)

        pth = os.path.dirname(name)
        prefix = name.replace(".meshb", "").replace(".mesh", "")
        config_fname = prefix + ".json"
        if os.path.exists(config_fname):
            logging.info(f"Config file: {config_fname}")
            with open(config_fname, "r") as f:
                config = json.load(f)
            if "names" in config:
                self._names = {tag: name for name, tag in config["names"].items()}
                logging.info(f"Names: {self._names}")
            if "steps" in config:
                config = config["steps"][time_step]

            self._node_sols = {}
            if "vertex_fields" in config:
                for var_name, sol_fname in config["vertex_fields"].items():
                    logging.info(f"Field {var_name}: {sol_fname}")
                    self._node_sols[var_name] = MeshbReader(
                        os.path.join(pth, sol_fname)
                    )

            self._cell_sols = {}
            if "cell_fields" in config:
                for var_name, sol_fname in config["cell_fields"].items():
                    logging.info(f"Field {var_name}: {sol_fname}")
                    self._cell_sols[var_name] = MeshbReader(
                        os.path.join(pth, sol_fname)
                    )

            self.Modified()

    def _get_timesteps(self):
        if len(self._file_names) > 1:
            return list(range(len(self._file_names)))
        else:
            return self._timesteps

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

    def _get_update_time(self, out_info):
        executive = self.GetExecutive()
        timesteps = self._get_timesteps()
        if timesteps is None or len(timesteps) == 0:
            return None
        elif out_info.Has(executive.UPDATE_TIME_STEP()) and len(timesteps) > 0:
            utime = out_info.Get(executive.UPDATE_TIME_STEP())
            dtime = timesteps[0]
            for atime in timesteps:
                if atime > utime:
                    return dtime
                else:
                    dtime = atime
            return dtime
        elif self._current_time is not None:
            return self._current_time
        else:
            assert len(timesteps) > 0
            return timesteps[0]

    def RequestInformation(self, request, inInfoVec, outInfoVec):

        executive = self.GetExecutive()
        outInfo = outInfoVec.GetInformationObject(0)
        outInfo.Remove(executive.TIME_STEPS())
        outInfo.Remove(executive.TIME_RANGE())
        timesteps = self._get_timesteps()
        assert len(timesteps) > 0

        for t in timesteps:
            outInfo.Append(executive.TIME_STEPS(), t)
        outInfo.Append(executive.TIME_RANGE(), timesteps[0])
        outInfo.Append(executive.TIME_RANGE(), timesteps[-1])

        self.__read_files_for_step(None)

        for var_name in self._node_sols.keys():
            self._arrayselection.AddArray(var_name)
            if self._first_load:
                self._arrayselection.EnableArray(var_name)
        for var_name in self._cell_sols.keys():
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
        cell_ids = []
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
                    cell_ids.append((etype, np.nonzero(flg)))

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

        return pug, used_verts, cell_ids

    def RequestData(self, request, inInfoVec, outInfoVec):

        outInfo = outInfoVec.GetInformationObject(0)
        time_step = self._get_update_time(outInfo)
        self.__read_files_for_step(time_step)

        xyz = self._mesh.read_vertices()
        if xyz.shape[1] == 2:
            xyz = np.hstack([xyz, np.zeros([xyz.shape[0], 1])])

        cells = self._mesh.read_elements()
        cell_dim = max([CELL_TYPES[etype][1] for etype in cells.keys()])

        node_data = {
            name: reader.read_sol(VERTEX) for name, reader in self._node_sols.items()
        }

        cell_data = {
            name: {
                etype: self._cell_sols[name].read_sol(etype) for etype in cells.keys()
            }
            for name in self._cell_sols.keys()
        }

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
                if dim == cell_dim:
                    name = "cell_tag_%d" % tag
                else:
                    try:
                        name = self._names[tag]
                    except:
                        name = "bdy_tag_%d" % tag
                (pug, vert_ids, cell_ids) = self._create_unstructured_grid(
                    dim, tag, xyz, cells
                )
                if pug is not None:
                    logging.info(
                        "%s: %d vertices, %d cells"
                        % (name, pug.GetNumberOfPoints(), pug.GetNumberOfCells())
                    )
                    mbds.SetBlock(iz, pug.VTKObject)
                    mbds.GetMetaData(iz).Set(mbds.NAME(), name)
                    iz += 1
                    data = pug.GetPointData()
                    added = []
                    for name, arr in node_data.items():
                        if name not in added:
                            data.append(arr[vert_ids], name)
                    data = pug.GetCellData()
                    added = []
                    if dim == cell_dim:
                        for name, arrays in cell_data.items():
                            if name not in added:
                                arr = []
                                for etype, ids in cell_ids:
                                    arr.append(arrays[etype][ids])
                                arr = np.concatenate(arr)
                                data.append(arr, name)

        return 1


def test(fname):
    from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter

    reader = PythonMeshbReader()
    reader.AddFileName(fname)
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

    test("mesh.meshb")
