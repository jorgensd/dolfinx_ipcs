
# Copyright (C) 2021-2022 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse
import os

import dolfinx.graph
import dolfinx.io
import gmsh  # With gmsh-nox-dev we have to import it before dolfinx
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

__all__ = ["markers"]
markers = {"Fluid": 1, "Inlet": 2, "Outlet": 3, "Walls": 4, "Obstacle": 5}


def generate_2D_channel(filename: str, outdir: str, res_min: float = 0.007, res_max: float = 0.05):
    """
    Generate mesh for benchmark DFG 2D-3:
    http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html

    Parameters
    ==========
    res_min
        Minimal mesh resolution around obstacle
    res_max
        Maximal mesh resolution at outlet
    """
    # Problem parameters
    L = 2.2
    H = 0.41
    c_x = c_y = 0.2
    r = 0.05
    gdim = 2

    gmsh.initialize()
    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        gmsh.model.addPhysicalGroup(
            volumes[0][0], [volumes[0][1]], markers["Fluid"])
        gmsh.model.setPhysicalName(volumes[0][0], markers["Fluid"], "Fluid")
        inflow, outflow, walls, obstacle = [], [], [], []
        boundaries = gmsh.model.getBoundary(volumes)
        for boundary in boundaries:
            # Bug in gmsh-nox-dev returns a -5 here
            if boundary[1] < 0:
                boundary = (boundary[0], abs(boundary[1]))
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, markers["Walls"])
        gmsh.model.setPhysicalName(1, markers["Walls"], "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, markers["Inlet"])
        gmsh.model.setPhysicalName(1, markers["Inlet"], "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, markers["Outlet"])
        gmsh.model.setPhysicalName(1, markers["Outlet"], "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, markers["Obstacle"])
        gmsh.model.setPhysicalName(1, markers["Obstacle"], "Obstacle")

        # Create distance field from obstacle.
        # Add threshold of mesh sizes based on the distance field
        # LcMax -                  /--------
        #                      /
        # LcMin -o---------/
        #        |         |       |
        #       Point    DistMin DistMax

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", obstacle)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 4 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 8 * r)

        # We take the minimum of the two fields as the mesh size
        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)

        gmsh.model.mesh.generate(gdim)

    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)

    gmsh.finalize()

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{outdir}/{filename}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{outdir}/{filename}_facets.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft, mesh.geometry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMSH scripts to generate the mesh for the DFG 2D-3 benchmark"
        + "http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--res-min", default=0.007, type=np.float64, dest="resmin",
                        help="Minimal mesh resolution (at obstacle)")
    parser.add_argument("--res-max", default=0.05, type=np.float64, dest="resmax",
                        help="Maximal mesh resolution (at outlet)")
    parser.add_argument("--filename", default="channel2D", type=str, dest="filename",
                        help="Name of output file (without XDMF extension)")
    parser.add_argument("--outdir", default="meshes", type=str, dest="outdir",
                        help="Name of output folder")
    args = parser.parse_args()
    os.system(f"mkdir -p {args.outdir}")
    generate_2D_channel(args.filename, args.outdir, res_min=args.resmin, res_max=args.resmax)
