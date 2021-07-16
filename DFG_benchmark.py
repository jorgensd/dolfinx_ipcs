import argparse
import os
import dolfinx
import dolfinx.io
import ufl
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

has_tqdm = True
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    has_tqdm = False
    print("To view progress with progressbar please install tqdm: `pip3 install tqdm`")

dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)


def IPCS(outdir: str, dim: int, degree_u: int,
         jit_parameters: dict = {}):
    #   {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]}):
    #     assert degree_u >= 2

    # Read in mesh
    comm = MPI.COMM_WORLD
    with dolfinx.io.XDMFFile(comm, f"meshes/channel{dim}D.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)

    with dolfinx.io.XDMFFile(comm, f"meshes/channel{dim}D_facets.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Facet tags")

    # Create output files
    out_u = dolfinx.io.XDMFFile(comm, f"{outdir}/u_{dim}D.xdmf", "w")
    out_u.write_mesh(mesh)
    out_p = dolfinx.io.XDMFFile(comm, f"{outdir}/p_{dim}D.xdmf", "w")
    out_p.write_mesh(mesh)

    # Define function spaces
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", degree_u))
    Q = dolfinx.FunctionSpace(mesh, ("CG", degree_u - 1))

    # Temporal parameters
    t = 0
    dt = 1e-2  # 5e-3
    T = 5 * dt  # 8

    # Physical parameters
    nu = 0.001
    f = dolfinx.Constant(mesh, (0,) * mesh.geometry.dim)
    H = 0.41
    Um = 2.25

    # Define functions for the variational form
    uh = dolfinx.Function(V)
    uh.name = "Velocity"
    u_tent = dolfinx.Function(V)
    u_tent.name = "Tentative_velocity"
    u_old = dolfinx.Function(V)
    ph = dolfinx.Function(Q)
    ph.name = "Pressure"
    phi = dolfinx.Function(Q)
    phi.name = "Phi"

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    # ----Step 1: Tentative velocity step----
    w_time = dolfinx.Constant(mesh, 3 / (2 * dt))
    w_diffusion = dolfinx.Constant(mesh, nu)
    a_tent = w_time * ufl.inner(u, v) * dx + w_diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v)) * dx
    L_tent += dolfinx.Constant(mesh, 1 / (2 * dt)) * ufl.inner(dolfinx.Constant(mesh, 4) * uh - u_old, v) * dx
    # BDF2 with implicit Adams-Bashforth
    bs = dolfinx.Constant(mesh, 2) * uh - u_old
    a_tent += ufl.inner(ufl.grad(u) * bs, v) * dx
    # Temam-device
    a_tent += dolfinx.Constant(mesh, 0.5) * ufl.div(bs) * ufl.inner(u, v) * dx

    # Find boundary facets and create boundary condition
    markers = {"Fluid": 1, "Inlet": 2, "Outlet": 3, "Walls": 4, "Obstacle": 5}
    inlet_facets = mt.indices[mt.values == markers["Inlet"]]
    inlet_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, inlet_facets)
    wall_facets = mt.indices[mt.values == markers["Walls"]]
    wall_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, wall_facets)
    obstacle_facets = mt.indices[mt.values == markers["Obstacle"]]
    obstacle_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, obstacle_facets)

    def inlet_velocity(t):
        if mesh.geometry.dim == 3:
            return lambda x: ((16 * np.sin(np.pi * t / T) * Um * x[1] * x[2] * (H - x[1]) * (H - x[2]) / (H**4),
                               np.zeros(x.shape[1]), np.zeros(x.shape[1])))
        elif mesh.geometry.dim == 2:
            U = 1.5 * np.sin(np.pi * t / T)
            return lambda x: np.row_stack((4 * U * x[1] * (0.41 - x[1]) / (0.41**2), np.zeros(x.shape[1])))

    u_inlet = dolfinx.Function(V)
    u_inlet.interpolate(inlet_velocity(t))
    u_zero = dolfinx.Function(V)
    u_zero.x.array[:] = 0.0

    bcs_tent = [dolfinx.DirichletBC(u_inlet, inlet_dofs), dolfinx.DirichletBC(
        u_zero, wall_dofs), dolfinx.DirichletBC(u_zero, obstacle_dofs)]
    a_tent = dolfinx.Form(a_tent, jit_parameters=jit_parameters)
    A_tent = dolfinx.fem.assemble_matrix(a_tent, bcs=bcs_tent)
    A_tent.assemble()
    L_tent = dolfinx.Form(L_tent, jit_parameters=jit_parameters)
    b_tent = dolfinx.Function(V)

    # Step 2: Pressure correction step
    outlet_facets = mt.indices[mt.values == markers["Outlet"]]
    outlet_dofs = dolfinx.fem.locate_dofs_topological(Q, fdim, outlet_facets)
    p_zero = dolfinx.Function(Q)
    p_zero.x.array[:] = 0
    bcs_corr = [dolfinx.DirichletBC(p_zero, outlet_dofs)]
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
    L_corr = - w_time * ufl.inner(ufl.div(u_tent), q) * dx
    a_corr = dolfinx.Form(a_corr, jit_parameters=jit_parameters)
    A_corr = dolfinx.fem.assemble_matrix(a_corr, bcs=bcs_corr)
    A_corr.assemble()

    b_corr = dolfinx.Function(Q)
    L_corr = dolfinx.Form(L_corr, jit_parameters=jit_parameters)

    # Step 3: Velocity update
    a_up = dolfinx.fem.Form(ufl.inner(u, v) * dx, jit_parameters=jit_parameters)
    L_up = dolfinx.fem.Form(ufl.inner(u_tent, v) * dx - w_time**(-1) * ufl.inner(ufl.grad(phi), v) * dx,
                            jit_parameters=jit_parameters)
    A_up = dolfinx.fem.assemble_matrix(a_up)
    A_up.assemble()
    b_up = dolfinx.Function(V)

    # Setup solvers
    rtol = 1e-8
    atol = 1e-8
    solver_tent = PETSc.KSP().create(comm)
    solver_tent.setOperators(A_tent)
    solver_tent.setTolerances(rtol=rtol, atol=atol)
    solver_tent.rtol = rtol
    # solver_tent.setType("bcgs")
    # solver_tent.getPC().setType("jacobi")
    solver_tent.setType("preonly")
    solver_tent.getPC().setType("lu")
    solver_tent.getPC().setFactorSolverType("mumps")

    solver_corr = PETSc.KSP().create(comm)
    solver_corr.setOperators(A_corr)
    solver_corr.setTolerances(rtol=rtol, atol=atol)
    solver_corr.setType("preonly")
    solver_corr.getPC().setType("lu")
    solver_corr.getPC().setFactorSolverType("mumps")
    # solver_corr.setInitialGuessNonzero(True)
    # solver_corr.max_it = 200
    # solver_corr.setType("gmres")
    # solver_corr.getPC().setType("hypre")
    # solver_corr.getPC().setHYPREType("boomeramg")

    solver_up = PETSc.KSP().create(comm)
    solver_up.setOperators(A_up)
    solver_up.setTolerances(rtol=rtol, atol=atol)
    solver_up.setType("preonly")
    solver_up.getPC().setType("lu")
    solver_up.getPC().setFactorSolverType("mumps")
    # solver_up.setInitialGuessNonzero(True)
    # solver_up.max_it = 200
    # solver_up.setType("cg")
    # solver_up.getPC().setType("jacobi")

    # Solve problem
    out_u.write_function(uh, t)
    out_u.write_function(u_tent, t)
    out_p.write_function(ph, t)
    N = int(T / dt)
    if has_tqdm:
        time_range = tqdm(range(N))
    else:
        time_range = range(N)
    for i in time_range:

        t += dt
        # Solve step 1
        with dolfinx.common.Timer("~Step 1"):
            u_inlet.interpolate(inlet_velocity(t))
            A_tent.zeroEntries()
            dolfinx.fem.assemble_matrix(A_tent, a_tent, bcs=bcs_tent)
            A_tent.assemble()
            b_tent.x.array[:] = 0
            dolfinx.fem.assemble_vector(b_tent.vector, L_tent)
            dolfinx.fem.assemble.apply_lifting(b_tent.vector, [a_tent], [bcs_tent])
            b_tent.x.scatter_reverse(dolfinx.cpp.common.ScatterMode.add)
            dolfinx.fem.assemble.set_bc(b_tent.vector, bcs_tent)
            solver_tent.solve(b_tent.vector, u_tent.vector)
            u_tent.x.scatter_forward()
            out_u.write_function(u_tent, t)

        # Solve step 2
        with dolfinx.common.Timer("~Step 2"):
            b_corr.x.array[:] = 0
            dolfinx.fem.assemble_vector(b_corr.vector, L_corr)
            dolfinx.fem.assemble.apply_lifting(b_corr.vector, [a_corr], [bcs_corr])
            b_corr.x.scatter_reverse(dolfinx.cpp.common.ScatterMode.add)
            dolfinx.fem.assemble.set_bc(b_corr.vector, bcs_corr)
            solver_corr.solve(b_corr.vector, phi.vector)
            phi.x.scatter_forward()

            # Update p and previous u
            ph.vector.axpy(1.0, phi.vector)
            ph.x.scatter_forward()
            u_old.x.array[:] = uh.x.array
            u_old.x.scatter_forward()

        # Solve step 3
        with dolfinx.common.Timer("~Step 3"):
            b_up.x.array[:] = 0
            dolfinx.fem.assemble_vector(b_up.vector, L_up)
            b_up.x.scatter_reverse(dolfinx.cpp.common.ScatterMode.add)
            solver_up.solve(b_up.vector, uh.vector)
            uh.x.scatter_forward()

        with dolfinx.common.Timer("~IO"):
            out_u.write_function(uh, t)
            out_p.write_function(ph, t)

        print(dolfinx.fem.assemble_scalar(ufl.inner(ufl.grad(phi), ufl.grad(phi))
              * ufl.ds(domain=mesh, subdomain_data=mt, subdomain_id=markers["Obstacle"])))

        print("flux", dolfinx.fem.assemble_scalar(ufl.inner(uh, uh)
              * ufl.ds(domain=mesh, subdomain_data=mt, subdomain_id=markers["Obstacle"])))

    out_u.close()
    out_p.close()
    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run the DFG 2D-3 benchmark"
        + "http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--degree-u", default=2, type=int, dest="degree",
                        help="Degree of velocity space")
    _2D = parser.add_mutually_exclusive_group(required=False)
    _2D.add_argument('--3D', dest='threed', action='store_true', help="Use 3D mesh", default=False)
    parser.add_argument("--outdir", default="results", type=str, dest="outdir",
                        help="Name of output folder")
    args = parser.parse_args()
    dim = 3 if args.threed else 2
    os.system(f"mkdir -p {args.outdir}")
    IPCS(args.outdir, dim=dim, degree_u=args.degree)
