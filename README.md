Repository for a simple BDF2 IPCS solver for the Navier-Stokes equations with an implicit Adam-Bashforth linearization and a Temam device.

# Instalation
To install the dependicies, docker is recommended:
```
docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared/ --rm dolfinx/dolfinx
```

To be able to use time progressbar functionality, please install tqdm
```bash
pip3 install tqdm
```

# Taylor-Green benchmark
To run the 2D Taylor-Green benchmark, run `python3 ipcs.py`.

# [DFG 3D benchmark](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d.html)
The resolution of the mesh can be changed by modifying  the lc variable in `cfd.geo`

To generate the mesh from the geo file, run
```bash
python3 create_and_convert_3D_mesh.py
```
The problem is solved by running
```bash
python3 dfg3d.py
```

There is also an option to solve the 2D turek problem, by generating the 2D mesh by running
```bash
python3 create_and_convert_2D_mesh.py
```
and changing `dim` from `2` to `3` as input to the `IPCS` class in `dfg3d.py`.