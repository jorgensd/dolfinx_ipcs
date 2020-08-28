import meshio
import numpy as np

msh = meshio.read("cfd.msh")
cells = np.vstack(np.array([cells.data for cells in msh.cells
 if cells.type == "tetra"]))
cell_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh.cell_data_dict["gmsh:physical"].keys()
                            if key == "tetra"])
mesh = meshio.Mesh(points=msh.points,
                       cells=[("tetra", cells)],
                       cell_data={"markers": cell_data})
meshio.xdmf.write("mesh.xdmf", mesh)


facets =  np.vstack(np.array([cells.data for cells in msh.cells
 if cells.type == "triangle"]))
facet_data = np.vstack([msh.cell_data_dict["gmsh:physical"][key]
                            for key in
                            msh.cell_data_dict["gmsh:physical"].keys()
                            if key == "triangle"])
facet_mesh = meshio.Mesh(points=msh.points,
                       cells=[("triangle", facets)],
                       cell_data={"markers": facet_data})
meshio.xdmf.write("facets.xdmf", facet_mesh)
