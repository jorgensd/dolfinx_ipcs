from IPython import embed
import dolfinx
import dolfinx.io
import ufl
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
def compute_l2_time_err(dt, errors): return np.sqrt(dt*sum(errors))


def compute_eoc(errors):
    return np.log(errors[:-1]/errors[1:])/np.log(2)


def IPCS(r_lvl, t_lvl, degree_u=2):
    # Define mesh and function spaces
    N = 10*2**r_lvl
    mesh = dolfinx.RectangleMesh(comm, [np.array([-1.0, -1.0, 0.0]),
                                        np.array([2.0, 2.0, 0.0])],
                                 [N, N], dolfinx.cpp.mesh.CellType.triangle)
    celldim = mesh.topology.dim
    facetdim = celldim - 1
    degree_p = degree_u - 1
    error_raise = 3
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", degree_u))
    Q = dolfinx.FunctionSpace(mesh, ("CG", degree_p))

    # Temporal parameters
    t = 0
    dt = 0.1*0.5**t_lvl
    T = 1

    # Physical parameters
    nu = 0.01
    f = dolfinx.Constant(mesh, (0, 0))

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

    def u_ex(t, nu):
        """
        Wrapper to generate a function for interpolating given any pair of t and nu
        for the analytical expression of u
        """
        return lambda x: np.row_stack((
            -np.cos(np.pi*x[0])*np.sin(np.pi*x[1])
            * np.exp(-2.0*nu*np.pi**2*t),
            np.cos(np.pi*x[1])*np.sin(np.pi*x[0])
            * np.exp(-2.0*nu*np.pi**2*t)))

    def p_ex(t, nu):
        """
        Wrapper to generate a function for interpolating given any pair of t and nu
        for the analytical expression of
        """
        return lambda x: -0.25*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))\
            * np.exp(-4.0*nu*np.pi**2*t)

    # Interpolate initial guesses
    uh.interpolate(u_ex(t, nu))
    u_old.interpolate(u_ex(t-dt, nu))
    ph.interpolate(p_ex(t, nu))

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Step 1: Tentative velocity step
    w_time = dolfinx.Constant(mesh, 3/(2*dt))
    w_diffusion = dolfinx.Constant(mesh, nu)
    a_tent = (w_time * ufl.inner(u, v) + w_diffusion *
              ufl.inner(ufl.grad(u), ufl.grad(v)))*ufl.dx
    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v))*ufl.dx
    L_tent += dolfinx.Constant(mesh, 1/(2*dt)) *\
        ufl.inner(dolfinx.Constant(mesh, 4)*uh-u_old, v)*ufl.dx
    # BDF2 with implicit Adams-Bashforth
    bs = dolfinx.Constant(mesh, 2)*uh - u_old
    a_tent += ufl.inner(ufl.grad(u)*bs, v)*ufl.dx
    # Temam-device
    a_tent += dolfinx.Constant(mesh, 0.5)*ufl.div(bs)*ufl.inner(u, v)*ufl.dx
    # Find boundary facets and create boundary condition
    mesh.topology.create_connectivity(facetdim, celldim)
    bndry_facets = np.where(np.array(
        dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    bdofsV = dolfinx.fem.locate_dofs_topological(V, facetdim, bndry_facets)
    u_bc = dolfinx.Function(V)
    u_bc.interpolate(u_ex(t+dt, nu))
    bcs_tent = [dolfinx.DirichletBC(u_bc, bdofsV)]
    A_tent = dolfinx.fem.assemble_matrix(a_tent, bcs=bcs_tent)
    A_tent.assemble()
    b_tent = dolfinx.fem.assemble_vector(L_tent)
    b_tent.assemble()

    # Step 2: Pressure correction step
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx
    L_corr = - w_time*ufl.inner(ufl.div(u_tent), q)*ufl.dx
    nullspace = PETSc.NullSpace().create(constant=True)
    A_corr = dolfinx.fem.assemble_matrix(a_corr)
    A_corr.setNullSpace(nullspace)
    A_corr.assemble()
    b_corr = dolfinx.fem.assemble_vector(L_corr)
    b_corr.assemble()

    # Step 3: Velocity update
    a_up = ufl.inner(u, v)*ufl.dx
    L_up = (ufl.inner(u_tent, v) - w_time**(-1)
            * ufl.inner(ufl.grad(phi), v))*ufl.dx
    A_up = dolfinx.fem.assemble_matrix(a_up)
    A_up.assemble()
    b_up = dolfinx.fem.assemble_vector(L_up)
    b_up.assemble()

    # Setup solvers
    solver_tent = PETSc.KSP().create(comm)
    solver_tent.setType("preonly")
    solver_tent.setTolerances(rtol=1.0e-14)
    solver_tent.getPC().setType("lu")
    solver_tent.getPC().setFactorSolverType("mumps")
    solver_tent.setOperators(A_tent)

    solver_corr = PETSc.KSP().create(comm)
    solver_corr.setType("preonly")
    solver_corr.setTolerances(rtol=1.0e-14)
    solver_corr.getPC().setType("lu")
    solver_corr.getPC().setFactorSolverType("mumps")
    solver_corr.setOperators(A_corr)

    solver_up = PETSc.KSP().create(comm)
    solver_up.setType("preonly")
    solver_up.setTolerances(rtol=1.0e-14)
    solver_up.getPC().setType("lu")
    solver_up.getPC().setFactorSolverType("mumps")
    solver_up.setOperators(A_up)

    # Create spaces for error approximation
    V_err = dolfinx.VectorFunctionSpace(mesh, ("CG", degree_u+error_raise))
    Q_err = dolfinx.FunctionSpace(mesh, ("CG", degree_p+error_raise))
    u_err = dolfinx.Function(V_err)
    p_err = dolfinx.Function(Q_err)

    # Create file for output
    outfile = dolfinx.io.XDMFFile(comm, "output.xdmf", "w")
    outfile.write_mesh(mesh)

    # Solve problem
    l2_u = np.zeros(int(T/dt), dtype=np.float64)
    l2_p = np.zeros(int(T/dt), dtype=np.float64)
    vol = mesh.mpi_comm().allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.Constant(mesh, 1)*ufl.dx),
        op=MPI.SUM)
    i = 0
    outfile.write_function(uh, t)
    outfile.write_function(ph, t)
    while(t <= T-1e-3):
        t += dt
        # Update BC and exact solutions
        u_bc.interpolate(u_ex(t, nu))
        u_err.interpolate(u_ex(t, nu))
        p_err.interpolate(p_ex(t, nu))

        # Solve step 1
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
        A_corr.zeroEntries()
        dolfinx.fem.assemble_matrix(A_corr, a_corr)
        A_corr.assemble()
        with b_corr.localForm() as b_local:
            b_local.set(0.0)

        dolfinx.fem.assemble_vector(b_corr, L_corr)
        b_corr.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        b_corr.assemble()
        solver_corr.solve(b_corr, phi.vector)
        phi.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)

        # Normalize pressure correction
        phi_avg = mesh.mpi_comm().allreduce(
            dolfinx.fem.assemble_scalar(phi*ufl.dx)/vol,
            op=MPI.SUM)
        avg_vec = phi.vector.copy()
        with avg_vec.localForm() as avg_local:
            avg_local.set(-phi_avg)
        phi.vector.axpy(1.0, avg_vec)
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
        A_up.zeroEntries()
        dolfinx.fem.assemble_matrix(A_up, a_up)
        A_up.assemble()
        with b_up.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b_up, L_up)
        b_up.ghostUpdate(addv=PETSc.InsertMode.ADD,
                         mode=PETSc.ScatterMode.REVERSE)
        solver_up.solve(b_up, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)

        # Compute L2 erorr norms
        uL2 = mesh.mpi_comm().allreduce(
            dolfinx.fem.assemble_scalar(ufl.inner(uh-u_err, uh-u_err)*ufl.dx),
            op=MPI.SUM)
        pL2 = mesh.mpi_comm().allreduce(
            dolfinx.fem.assemble_scalar(ufl.inner(ph-p_err, ph-p_err)*ufl.dx),
            op=MPI.SUM)
        l2_u[i] = uL2
        l2_p[i] = pL2

        i += 1
        outfile.write_function(phi, t)
        outfile.write_function(u_tent, t)
        outfile.write_function(uh, t)
        outfile.write_function(ph, t)
    outfile.close()
    L2L2u = compute_l2_time_err(dt, l2_u)
    L2L2p = compute_l2_time_err(dt, l2_p)
    return L2L2u, L2L2p


if __name__ == "__main__":
    R_ref = 5
    T_ref = 5
    errors_u = np.zeros((R_ref, T_ref), dtype=np.float64)
    errors_p = np.zeros((R_ref, T_ref), dtype=np.float64)
    for i in range(R_ref):
        for j in range(T_ref):
            errors_u[i, j], errors_p[i, j] = IPCS(i, j, degree_u=2)
            print(i, j, errors_u[i, j], errors_p[i, j])

    print("Temporal eoc u", compute_eoc(errors_u[-1, :]))
    print("Spatial eoc u", compute_eoc(errors_u[:, -1]))
    print("Temporal eoc p", compute_eoc(errors_p[-1, :]))
    print("Spatial eoc p", compute_eoc(errors_p[:, -1]))
