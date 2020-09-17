import meshio
import numpy as np
import os
os.system("gmsh -3 -optimize_netgen cfd.geo")

msh = meshio.read("cfd.msh")
cells = np.vstack(np.array([cells.data for cells in msh.cells
 if cells.type == "tetra10"]))
cell_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh.cell_data_dict["gmsh:physical"].keys()
                            if key == "tetra10"])
mesh = meshio.Mesh(points=msh.points,
                       cells=[("tetra10", cells)],
                       cell_data={"markers": cell_data})
meshio.xdmf.write("mesh.xdmf", mesh)


facets =  np.vstack(np.array([cells.data for cells in msh.cells
 if cells.type == "triangle6"]))
facet_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh.cell_data_dict["gmsh:physical"].keys()
                            if key == "triangle6"])
facet_mesh = meshio.Mesh(points=msh.points,
                       cells=[("triangle6", facets)],
                       cell_data={"markers": facet_data})
meshio.xdmf.write("facets.xdmf", facet_mesh)
