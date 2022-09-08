[![DOLFINx testing](https://github.com/jorgensd/dolfinx_ipcs/actions/workflows/testing.yml/badge.svg)](https://github.com/jorgensd/dolfinx_ipcs/actions/workflows/testing.yml)

Author: Jørgen S. Dokken

Repository for a simple BDF2 IPCS solver for the Navier-Stokes equations with an implicit Adam-Bashforth linearization and a Temam device.

# Installation
The code in this repository require [GMSH](https://gmsh.info/), including Python interface, [DOLFINx](https://github.com/FEniCS/dolfinx/) and [tqdm](https://github.com/tqdm/tqdm).

There are various ways to install these packages.

## Docker

You can use the pre-build docker image `ghcr.io/jorgensd/dolfinx_ipcs:v0.5.0` from [here](https://github.com/jorgensd/dolfinx_ipcs/pkgs/container/dolfinx_ipcs)

# Taylor-Green benchmark
To run the 2D Taylor-Green benchmark, run `ipcs.py`.
Use
```bash
python3 ipcs.py --help
``` 
for command-line options.

# [DFG 3D benchmark](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d.html)
The resolution of the mesh can be changed by modifying  the lc variable in `cfd.geo`

To generate the mesh from the geo file, run
```bash
python3 create_and_convert_3D_mesh.py
```
Run 
```bash
python3 create_and_convert_3D_mesh.py --help
```
for command-line options.

To solve the problem, run
```bash
python3 DFG_benchmark.py --3D
```

# [DFG 2D-3](http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html)
To generate a mesh for the 2D problem, run 
```bash
python3 create_and_convert_2D_mesh.py --help
```
for command-line options.

The problem is solved by running
```bash
python3 DFG_benchmark.py
```
