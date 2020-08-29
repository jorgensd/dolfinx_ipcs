Repository for a simple BDF2 IPCS solver for the Navier-Stokes equations with an implicit Adam-Bashforth linearization and a Temam device.

To install the dependicies, docker is recommended:
```
docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared/ --rm dolfinx/dolfinx
```


To run the 2D Taylor-Green benchmark, run `python3 ipcs.py`.

To run the [DFG 3D benchmark](http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d.html), run `python3 convert_mesh.py` followed by `python3 dfg3d.py`.
The resolution of the mesh can be changed by modifying  the lc variable in `cfd.geo`, recreate the msh file by `gmsh -3 cfd.geo` and rerun `python3 convert_mesh.py`.
