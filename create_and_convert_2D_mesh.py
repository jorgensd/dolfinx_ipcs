
# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse
import os

import dolfinx.io
import gmsh
import numpy as np
from mpi4py import MPI

__all__ = ["markers"]
markers = {"Fluid": 1, "Inlet": 2, "Outlet": 3, "Walls": 4, "Obstacle": 5}


def generate_2D_channel(filename: str, outdir: str, res_min: float = 0.01, res_max: float = 0.05):
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

    rank = MPI.COMM_WORLD.rank
    if rank == 0:
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

        # Get mesh geometry
        x = dolfinx.io.extract_gmsh_geometry(gmsh.model)

        # Get mesh topology for each element
        topologies = dolfinx.io.extract_gmsh_topology_and_markers(gmsh.model)
        # Get information about each cell type from the msh files
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
        for i, element in enumerate(topologies.keys()):
            properties = gmsh.model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, local_coords, _ = properties
            cell_information[i] = {"id": element, "dim": dim, "num_nodes": num_nodes}
            cell_dimensions[i] = dim

        # Sort elements by ascending dimension
        perm_sort = np.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)
        if tdim - 1 in cell_dimensions:
            num_facet_nodes = MPI.COMM_WORLD.bcast(cell_information[perm_sort[-2]]["num_nodes"], root=0)
            gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
            marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
            facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)

        cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
        # cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)

    else:
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = np.empty([0, num_nodes], np.int64), np.empty([0, gdim])
        # cell_values = np.empty((0,), dtype=np.int32)
        num_facet_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        marked_facets = np.empty((0, num_facet_nodes), dtype=np.int64)
        facet_values = np.empty((0,), dtype=np.int32)

    gmsh.finalize()

    # Create distributed mesh
    ufl_domain = dolfinx.io.ufl_mesh_from_gmsh(cell_id, gdim)
    gmsh_cell_perm = dolfinx.cpp.io.perm_gmsh(dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm]
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    # Permute facets from MSH to DOLFINx ordering
    facet_type = dolfinx.cpp.mesh.cell_entity_type(dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), fdim)
    gmsh_facet_perm = dolfinx.cpp.io.perm_gmsh(facet_type, num_facet_nodes)
    marked_facets = np.asarray(marked_facets[:, gmsh_facet_perm], dtype=np.int64)

    local_entities, local_values = dolfinx.cpp.io.extract_local_entities(mesh, fdim, marked_facets, facet_values)
    mesh.topology.create_connectivity(fdim, tdim)
    adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)

    # Create DOLFINx MeshTags
    ft = dolfinx.mesh.create_meshtags(mesh, fdim, adj, np.int32(local_values))
    ft.name = "Facet tags"

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{outdir}/{filename}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{outdir}/{filename}_facets.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMSH scripts to generate the mesh for the DFG 2D-3 benchmark"
        + "http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--res-min", default=0.01, type=np.float64, dest="resmin",
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
