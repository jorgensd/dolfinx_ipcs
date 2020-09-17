import dolfinx
import dolfinx.io
import ufl
import numpy as np
    
from mpi4py import MPI
from petsc4py import PETSc

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    if MPI.COMM_WORLD.rank == 0:
        print("please install tqdm: `pip3 install tqdm`")
    exit(1)


def IPCS(dim=3, degree_u=2):
    # Read in mesh
    if dim == 2:
        ext = "2D"
    else:
        ext = ""
    comm = MPI.COMM_WORLD
    with dolfinx.io.XDMFFile(comm, "mesh{0:s}.xdmf".format(ext), "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)

    out_u = dolfinx.io.VTKFile("results/u.pvd")
    out_p = dolfinx.io.VTKFile("results/p.pvd")

    with dolfinx.io.XDMFFile(
       comm, "facets{0:s}.xdmf".format(ext), "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Grid")
    # Define function spaces
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", degree_u))
    Q = dolfinx.FunctionSpace(mesh, ("CG", degree_u-1))
    # Temporal parameters
    t = 0
    dt = 5e-3
    T = 8

    # Physical parameters
    nu = 0.001
    f = dolfinx.Constant(mesh, (0,)*mesh.geometry.dim)
    H = 0.41
    Um = 2.25

    # Define functions for the variational form
    uh = dolfinx.Function(V)
    uh.name = "Velocity"
    u_tent = dolfinx.Function(V)
    u_tent.name = "Tentative velocity"
    u_old = dolfinx.Function(V)
    ph = dolfinx.Function(Q)
    ph.name = "Pressure"
    phi = dolfinx.Function(Q)
    phi.name = "Phi"

    # Define variational forms
    w_time = dolfinx.Constant(mesh, 3/(2*dt))
    w_diffusion = dolfinx.Constant(mesh, nu)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bs = 2*uh - u_old
    # Step 1: Tentative velocity step
    a_tent = (w_time * ufl.inner(u, v) + w_diffusion *
              ufl.inner(ufl.grad(u), ufl.grad(v)))*ufl.dx

    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v))*ufl.dx

    L_tent += dolfinx.Constant(mesh, 1/(2*dt))\
        * ufl.inner(4*uh-u_old, v)*ufl.dx

    # BDF2 with implicit Adams-Bashforth
    a_tent += ufl.inner(ufl.grad(u)*bs, v)*ufl.dx
    # Temam-device
    a_tent += dolfinx.Constant(mesh, 0.5)*ufl.div(bs)*ufl.inner(u, v)*ufl.dx

    # Find boundary facets and create boundary condition
    inlet_facets = mt.indices[mt.values == 1]
    inlet_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, inlet_facets)
    wall_facets = mt.indices[mt.values == 3]
    wall_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, wall_facets)
    obstacle_facets = mt.indices[mt.values == 4]
    obstacle_dofs = dolfinx.fem.locate_dofs_topological(
        V, fdim, obstacle_facets)

    def inlet_velocity(t):
        if mesh.geometry.dim == 3:
            return lambda x: np.row_stack(
            (16*np.sin(np.pi*t/T)*Um*x[1]*x[2]*(H-x[1])*(H-x[2])/(H**4),
             np.zeros(x.shape[1]), np.zeros(x.shape[1])))
        elif mesh.geometry.dim ==2:
              U = 1.5*np.sin(np.pi*t/T)
              return lambda x: np.row_stack(
                  (4*U*x[1]*(0.41-x[1])/(0.41**2), np.zeros(x.shape[1])))

    u_inlet = dolfinx.Function(V)
    u_zero = dolfinx.Function(V)
    with u_zero.vector.localForm() as u_local:
        u_local.set(0.0)

    bcs_tent = [dolfinx.DirichletBC(u_inlet, inlet_dofs),
                dolfinx.DirichletBC(u_zero, wall_dofs),
                dolfinx.DirichletBC(u_zero, obstacle_dofs)]

    A_tent = dolfinx.fem.assemble_matrix(a_tent, bcs=bcs_tent)
    A_tent.assemble()
    b_tent = dolfinx.fem.assemble_vector(L_tent)
    dolfinx.fem.assemble_vector(b_tent, L_tent)
    dolfinx.fem.assemble.apply_lifting(b_tent, [a_tent], [bcs_tent])
    b_tent.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.assemble.set_bc(b_tent, bcs_tent)

    # Step 2: Pressure correction step
    outlet_facets = mt.indices[mt.values == 2]
    outlet_dofs = dolfinx.fem.locate_dofs_topological(Q, fdim, outlet_facets)
    p_zero = dolfinx.Function(Q)
    with p_zero.vector.localForm() as p_local:
        p_local.set(0.0)
    bcs_corr = [dolfinx.DirichletBC(p_zero, outlet_dofs)]
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx
    L_corr = - w_time*ufl.inner(ufl.div(u_tent), q)*ufl.dx
    A_corr = dolfinx.fem.assemble_matrix(a_corr, bcs=bcs_corr)
    A_corr.assemble()

    b_corr = dolfinx.fem.assemble_vector(L_corr)
    dolfinx.fem.assemble.apply_lifting(b_corr, [a_corr], [bcs_corr])
    b_corr.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.assemble.set_bc(b_corr, bcs_corr)

    # Step 3: Velocity update
    a_up = ufl.inner(u, v)*ufl.dx
    L_up = (ufl.inner(u_tent, v)
            - 1/w_time * ufl.inner(ufl.grad(phi), v))*ufl.dx
    A_up = dolfinx.fem.assemble_matrix(a_up)
    A_up.assemble()
    b_up = dolfinx.fem.assemble_vector(L_up)
    b_up.assemble()

    # Setup solvers
    solver_tent = PETSc.KSP().create(comm)
    solver_tent.setOperators(A_tent)
    solver_tent.rtol = 1e-10
    solver_tent.setType("bcgs")
    solver_tent.getPC().setType("jacobi")

    solver_corr = PETSc.KSP().create(comm)
    solver_corr.setOperators(A_corr)
    solver_corr.rtol = 1e-10
    solver_corr.setType("cg")
    solver_corr.getPC().setType("gamg")
    solver_corr.getPC().setGAMGType("agg")

    solver_up = PETSc.KSP().create(comm)
    solver_up.setOperators(A_up)
    solver_up.setTolerances(rtol=1.0e-10)
    solver_up.setType("cg")
    solver_up.getPC().setType("gamg")
    solver_up.getPC().setGAMGType("agg")



    # Solve problem
    out_u.write(uh, t)
    out_p.write(ph, t)
    N = int(T/dt)
    for i in tqdm(range(N)):

        t += dt
        # Solve step 1
        u_inlet.interpolate(inlet_velocity(t))
        A_tent.zeroEntries()
        dolfinx.fem.assemble_matrix(A_tent, a_tent, bcs=bcs_tent)
        A_tent.assemble()
        with b_tent.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b_tent, L_tent)
        dolfinx.fem.assemble.apply_lifting(b_tent, [a_tent], [bcs_tent])
        b_tent.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.assemble.set_bc(b_tent, bcs_tent)
        solver_tent.solve(b_tent, u_tent.vector)
        u_tent.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)

        # Solve step 2
        with b_corr.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b_corr, L_corr)
        dolfinx.fem.assemble.apply_lifting(b_corr, [a_corr], [bcs_corr])
        b_corr.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.assemble.set_bc(b_corr, bcs_corr)

        solver_corr.solve(b_corr, phi.vector)
        phi.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)

        # Update p and previous u
        ph.vector.axpy(1.0, phi.vector)
        ph.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
        uh.vector.copy(result=u_old.vector)
        u_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                 mode=PETSc.ScatterMode.FORWARD)
        # Solve step 3
        with b_up.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b_up, L_up)
        b_up.ghostUpdate(addv=PETSc.InsertMode.ADD,
                         mode=PETSc.ScatterMode.REVERSE)
        solver_up.solve(b_up, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
    
        out_u.write(uh, t)
        out_p.write(ph, t)

        
if __name__ == "__main__":
    IPCS(dim=3, degree_u=2)
