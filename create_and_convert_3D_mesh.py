import meshio
import numpy as np
import os
os.system("gmsh -3 -optimize_netgen cfd.geo")

msh = meshio.read("cfd.msh")
try:
    celltype = "tetra10"
    facettype = "triangle6"
    cells = np.vstack(np.array([cells.data for cells in msh.cells
                      if cells.type == celltype]))
except ValueError:
    celltype = "tetra"
    facettype = "triangle"
    cells = np.vstack(np.array([cells.data for cells in msh.cells
                      if cells.type == celltype]))

cell_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                       for key in
                       msh.cell_data_dict["gmsh:physical"].keys()
                       if key == celltype])
mesh = meshio.Mesh(points=msh.points,
                   cells=[(celltype, cells)],
                   cell_data={"markers": cell_data})
meshio.xdmf.write("mesh.xdmf", mesh)


facets = np.vstack(np.array([cells.data for cells in msh.cells
                             if cells.type == facettype]))
facet_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                        for key in
                        msh.cell_data_dict["gmsh:physical"].keys()
                        if key == facettype])
facet_mesh = meshio.Mesh(points=msh.points,
                         cells=[(facettype, facets)],
                         cell_data={"markers": facet_data})
meshio.xdmf.write("facets.xdmf", facet_mesh)
