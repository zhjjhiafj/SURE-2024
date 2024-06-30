#Nonlinear Poisson Equation from FEniCSx tutorial
import ufl
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
def q(u):
    return 1 + u**2


domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]
f = - ufl.div(q(u_ufl) * ufl.grad(u_ufl))

V = fem.functionspace(domain, ("Lagrange", 1))
def u_exact(x): return eval(str(u_ufl))

u_D = fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

problem = NonlinearProblem(F, uh, bcs=[bc])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(uh)
assert (converged)
print(f"Number of interations: {n:d}")

# Compute L2 error and error at nodes
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_local = fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx))
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")

import pyvista
print(pyvista.global_theme.jupyter_backend)
from dolfinx import plot
pyvista.start_xvfb()
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
figure = u_plotter.screenshot("/home/zhenyu/SURE2024/6.27/nonlinear_poisson.png")
