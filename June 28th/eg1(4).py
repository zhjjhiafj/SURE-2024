#eg_1 build in solver
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, io, nls, log
import numpy as np
import pyvista
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
n=2.2
def q(u):
    return (((n/(2.0*n+2.0))**n) / (n + 2.0)) * (abs(ufl.dot(ufl.grad(u), ufl.grad(u)))) ** ((n - 1.0)/2)

num_intervals = 1000
start_point = -1.0
end_point = 1.0
domain = mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])
x = ufl.SpatialCoordinate(domain)
from dolfinx import default_scalar_type
f =fem.Constant(domain, default_scalar_type(1.0))

V = fem.functionspace(domain, ("Lagrange", 1))

u_D = dolfinx.fem.Function(V)
u_D.interpolate(lambda x: np.zeros_like(x[0]))
u_exact = dolfinx.fem.Function(V)
u_exact.interpolate(lambda x: 2.0* (n + 2.0) ** (1.0 / n) * (1.0 - abs(x[0]) ** (1.0 + 1.0 / n)))
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = fem.Function(V)
uh.interpolate(lambda x: 1-x[0]**2)
#uh.interpolate(lambda x: 2.0* (n + 2.0) ** (1.0 / n) * (1.0 - abs(x[0]) ** (1.0 + 1.0 / n)))
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
opts[f"{option_prefix}pc_type"] = "ilu"
#opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)
k, converged = solver.solve(uh)
assert (converged)
print(f"Number of interations: {k:d}")

# Compute L2 error and error at nodes
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_local = fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx))
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

import matplotlib.pyplot as plt

error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array - u_exact.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
print((1/(2.0+2.0)))
u_values = uh.x.array**(n/(2*n+2.0))
x_values = np.linspace(start_point, end_point, num_intervals + 1)
if domain.comm.rank == 0:
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(x_values, u_values, label="Numerical solution")
    #plt.plot(x_values, 6.0**0.25*(1.0-((x_values**2.0)**0.5)**2.0)**0.25, label="Exact solution", linestyle='dashed')
    plt.plot(x_values, (2 * (n + 2) ** (1.0 / n) * (1 - abs(x_values) ** (1.0 + 1.0 / n)))**(n/(2*n+2.0)), label="Exact solution",
             linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
   # plt.savefig("/home/zhenyu/SURE2024/7.10/test1.png")
    plt.show()
