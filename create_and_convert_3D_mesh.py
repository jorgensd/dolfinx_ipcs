# Copyright (C) 2021-2022 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse
import os

import dolfinx
import dolfinx.graph
from dolfinx.io import gmshio
from mpi4py import MPI


def generate_3D_channel(filename: str, outdir: str):
    gdim = 3
    os.system(f"gmsh -{gdim} -optimize_netgen cfd.geo")

    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, _, ft = gmshio.read_from_msh("cfd.msh", mesh_comm, model_rank)
    ft.name = "Facet tags"

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{outdir}/{filename}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{outdir}/{filename}_facets.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMSH scripts to generate the mesh in cfd.geo and convert it to DOLFINx format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filename", default="channel3D", type=str, dest="filename",
                        help="Name of output file (without XDMF extension)")
    parser.add_argument("--outdir", default="meshes", type=str, dest="outdir",
                        help="Name of output folder")
    args = parser.parse_args()
    os.system(f"mkdir -p {args.outdir}")
    generate_3D_channel(args.filename, args.outdir)
