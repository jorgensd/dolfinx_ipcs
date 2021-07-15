# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse
import os

import dolfinx
import dolfinx.io
import gmsh
import numpy as np
from mpi4py import MPI


def generate_3D_channel(filename: str):
    gdim = 3
    os.system("gmsh -{gdim} -optimize_netgen cfd.geo")
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.open("cfd.msh")

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

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{filename}_facets.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMSH scripts to generate the mesh in cfd.geo and convert it to DOLFINx format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filename", default="channel3D", type=str, dest="filename",
                        help="Name of output file (without XDMF extension)")
    args = parser.parse_args
    generate_3D_channel(args.filename)
