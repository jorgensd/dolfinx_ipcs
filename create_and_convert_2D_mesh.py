from pygmsh import generate_mesh
from pygmsh.built_in.geometry import Geometry
import meshio
import numpy as np
inflow = 1
outflow = 2
walls = 3
obstacle = 4

L = 2.2
H = 0.41
c_x,c_y = 0.2,0.2
r_x = 0.05
d = 0.1

def channel_mesh(res=0.025):
    """ 
    Creates a unstructured rectangular mesh
    """
    geometry = Geometry()

    c = geometry.add_point((c_x,c_y,0))    

    # Elliptic obstacle
    p1 = geometry.add_point((c_x-r_x, c_y,0),lcar=0.5*res)
    p2 = geometry.add_point((c_x, c_y+r_x,0),lcar=0.5*res)
    p3 = geometry.add_point((c_x+r_x, c_y,0),lcar=0.5*res)
    p4 = geometry.add_point((c_x, c_y-r_x,0),lcar=0.5*res)
    arc_1 = geometry.add_ellipse_arc(p1, c, p2, p2)
    arc_2 = geometry.add_ellipse_arc(p2, c, p3, p3)
    arc_3 = geometry.add_ellipse_arc(p3, c, p4, p4)
    arc_4 = geometry.add_ellipse_arc(p4, c, p1, p1)
    obstacle_loop = geometry.add_line_loop([arc_1, arc_2, arc_3, arc_4])
    
    geometry.add_physical_line(obstacle_loop.lines, label=obstacle)
    # Outer boundary
    points = [geometry.add_point((0, 0, 0),lcar=res),
              geometry.add_point((L, 0, 0),lcar=5*res),
              geometry.add_point((L, H, 0),lcar=5*res),
              geometry.add_point((0, H, 0),lcar=res)]
    lines = [geometry.add_line(points[i], points[i+1])
             for i in range(len(points)-1)]
    lines.append(geometry.add_line(points[-1], points[0]))
    ll = geometry.add_line_loop(lines)
    s = geometry.add_plane_surface(ll, holes=[obstacle_loop])
    outflow_list = [lines[1]]
    flow_list = [lines[-1]]
    wall_list = [lines[0], lines[2]]

    geometry.add_physical_surface(s,label=12)
    geometry.add_physical_line(flow_list, label=inflow)
    geometry.add_physical_line(outflow_list, label=outflow)
    geometry.add_physical_line(wall_list, label=walls)
    

    mesh = generate_mesh(geometry, prune_z_0=True)
    points = mesh.points
    cell_data = np.vstack([mesh.cell_data_dict["gmsh:physical"][key]
                       for key in
                       mesh.cell_data_dict["gmsh:physical"].keys()
                       if key == "triangle"])
    cells = np.vstack([cells.data for cells in mesh.cells
                   if cells.type == "triangle"])

    line_data = np.vstack([mesh.cell_data_dict["gmsh:physical"][key]
                           for key in
                           mesh.cell_data_dict["gmsh:physical"].keys()
                           if key == "line"])
    line_cells = np.vstack([cells.data for cells in mesh.cells
                            if cells.type == "line"])
    
    meshio.write("mesh2D.xdmf", meshio.Mesh(
        points=points, cells=[("triangle", cells)]))
    
    meshio.write("facets2D.xdmf", meshio.Mesh(
        points=points, cells=[("line", line_cells)],
        cell_data={"name_to_read":
                            line_data}))
        
if __name__=="__main__":
    import sys
    try:
        res = float(sys.argv[1])
    except IndexError:
        res = 0.05
    channel_mesh(0.01)
