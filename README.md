# Paraview reader for `.meshb` files

## Usage

### Minimal Paraview version

5.13

### Enable the plugin

In order to load polygons / polyhedra fast, build the `ctypes` extension using `make -C meshb_io` (edit the `Makefile` if needed).
[`libMeshb`](https://github.com/LoicMarechal/libMeshb) is copied here for simplicity

To use this reader in Paraview load `pvpymeshbreader.py` as a plugin using the Plugin Manager from  `Tools > Plugin Manager`. Enable `Auto Load`. 

### Using tag names

Names are not possibles in `.meshb` files. They can be given in a separate `json` file with the same prefix, e.g. `quadratic.json` for a mesh file `quadratic.meshb`:
```json
{
  "names": {
    "xmin": 1,
    "ymin": 2,
    "zmax": 3,
    "ymax": 4,
    "zmin": 5,
    "sphere": 6,
    "xmax": 7
  }
}
```

### Viewing curved elements

By default, Paraview uses 1 element subdivision for quadratic elements. To have a smoother representation, you can increase `Nonlinear Subdivision Level`.

### Viewing fields

TODO