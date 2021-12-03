[![DOLFINx testing](https://github.com/jorgensd/dolfinx_ipcs/actions/workflows/testing.yml/badge.svg)](https://github.com/jorgensd/dolfinx_ipcs/actions/workflows/testing.yml)

Author: Jørgen S. Dokken

Repository for a simple BDF2 IPCS solver for the Navier-Stokes equations with an implicit Adam-Bashforth linearization and a Temam device.

# Installation
The code in this repository require [GMSH](https://gmsh.info/), including Python interface, [DOLFINx](https://github.com/FEniCS/dolfinx/) and [tqdm](https://github.com/tqdm/tqdm).

There are various ways to install these packages.

## Docker

To install the dependicies, docker is recommended:
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm dolfinx/dolfinx
```

To be able to use time progressbar functionality, please install tqdm
```bash
pip3 install tqdm
```

## Pypi/requirements.txt

> :warning: This is currently experimental, and cannot be guaranteed to work.  Currently the meshes can be created and converted using this strategy, but the dolfinx installation is out of date.

The `requirement.txt` file in this repository can install both DOLFINx and GMSH (without GUI) using the command
```bash
pip3 install -r requirements.txt
```
The only requirement is that your system has an installation of `mpich` and `python3` (>=3.7).
A minimal docker environment based on ubuntu 20.04 is listed below
```docker
FROM ubuntu:20.04
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    git \
    wget \
    libmpich-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git clone https://github.com/jorgensd/dolfinx_ipcs/ && \
    pip3 install -r dolfinx_ipcs.py requirements.txt
```
which can also be found in the `Dockerfile`.

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
