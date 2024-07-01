#Modified from example from tutorial： Poisson equation 1D

from mpi4py import MPI
from dolfinx import mesh
import numpy as np
num_intervals = 10000
start_point = 0
end_point = 1.0
domain = mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])
from dolfinx import fem
from dolfinx.fem import functionspace
V = functionspace(domain, ("Lagrange", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: np.zeros_like(x[0]))

import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(1))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: -0.5*x[0]**2 + 0.5 * x[0])

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

import matplotlib.pyplot as plt
u_values = uh.x.array
x_values = np.linspace(start_point, end_point, num_intervals + 1)

if domain.comm.rank == 0:
    plt.figure()
    plt.plot(x_values, u_values, label="Numerical solution")
    plt.plot(x_values, -0.5 * x_values ** 2 + 0.5 * x_values, label="Exact solution", linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
    plt.savefig("/home/zhenyu/SURE2024/6.27/poisson_interval_solution.png")
    plt.show()

#from dolfinx import io
#from pathlib import Path

#results_folder = Path("/home/zhenyu/SURE2024/6.27/results")
#results_folder.mkdir(exist_ok=True, parents=True)
#filename = results_folder / "fundamentals"

#with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
#    vtx.write(0.0)

#with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
#    xdmf.write_mesh(domain)
#    xdmf.write_function(uh)
