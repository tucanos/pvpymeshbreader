import gmsh
import numpy as np
from meshb_io import MeshbWriter, MeshbReader, TRIANGLE, TRIANGLE2


def build_geom():
    """Create a 3D geometry and apply physical tags"""

    model = gmsh.model()
    model.add("Geometry")
    model.setCurrent("Geometry")
    sphere = model.occ.addSphere(0, 0, 0, 0.25, tag=1)
    box = model.occ.addBox(0, 0, 0, 1, 1, 1)
    geom = model.occ.cut([(3, box)], [(3, sphere)])

    model.occ.synchronize()

    tol = 1e-8
    names = {}
    for dim, tag in gmsh.model.getEntities(dim=2):
        mass = gmsh.model.occ.getMass(dim, tag)
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, tag)
        phy_tag = model.addPhysicalGroup(dim, [tag])
        if abs(x) < tol:
            name = "xmin"
        elif abs(x - 1) < tol:
            name = "xmax"
        elif abs(y) < tol:
            name = "ymin"
        elif abs(y - 1) < tol:
            name = "ymax"
        elif abs(z) < tol:
            name = "zmin"
        elif abs(z - 1) < tol:
            name = "zmax"
        else:
            name = "sphere"
        names[name] = phy_tag

    return model, names


def get_surface_mesh(model, order):
    """Get the surface mesh as numpy arrays"""
    model.mesh.generate(dim=2)
    model.mesh.set_order(order)

    ids, nodes, _ = model.mesh.get_nodes()
    assert np.array_equal(ids, np.arange(ids.size) + 1)
    nodes = nodes.reshape((-1, 3))

    elements = {}
    etypes = model.mesh.get_element_types()
    for etype in etypes:
        name, dim, order, m, _, _ = model.mesh.getElementProperties(etype)
        conn = []
        tags = []
        if dim != 2:
            continue
        for _, tag in model.getEntities(dim=dim):
            phy_tag = model.get_physical_groups_for_entity(dim, tag)
            assert len(phy_tag) == 1
            phy_tag = phy_tag[0]

            _, tmp = model.mesh.get_elements_by_type(etype, tag=tag)
            tmp = tmp.reshape((-1, m))
            conn.append(tmp)
            tags.append(phy_tag + np.zeros(tmp.shape[0], dtype=type(phy_tag)))

        conn = np.vstack(conn) - 1
        tags = np.concatenate(tags)
        elements[name] = (conn, tags)

    return nodes, elements


def write_meshb(model, order, fname):

    nodes, elements = get_surface_mesh(model, order)
    writer = MeshbWriter(fname, dim=3)
    writer.write_vertices(nodes)
    for etype, (conn, tags) in elements.items():
        if etype == "Triangle 3":
            writer.write_elements(TRIANGLE, conn, tags)
        elif etype == "Triangle 6":
            writer.write_elements(TRIANGLE2, conn, tags)
        else:
            raise NotImplementedError(f"{etype}")


if __name__ == "__main__":

    import json

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    model, names = build_geom()

    fname = "quadratic.meshb"
    config = {"mesh": fname, "names": names}

    write_meshb(model, order=2, fname="quadratic.meshb")

    with open("test.json", "w") as f:
        json.dump(config, f, indent=2)
