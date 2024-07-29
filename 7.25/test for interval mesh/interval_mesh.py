# Import required libraries
import matplotlib.pyplot as plt
from petsc4py import PETSc
import dolfinx
from dolfinx.fem.petsc import create_matrix
from dolfinx.la import create_petsc_vector
import ufl
import numpy
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle
from dolfinx.io import XDMFFile, gmshio
import gmsh
from dolfinx import default_scalar_type
num_intervals = 1000
start_point = -1.0
end_point = 1.0
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])
x = ufl.SpatialCoordinate(mesh)

V = fem.functionspace(mesh, ("CG", 1))

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(-0.00001)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(np.infty)
fdim = mesh.topology.dim - 1
u_D = fem.Function(V)
u_D.interpolate(lambda x: np.zeros_like(x[0]))
boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))
u=fem.Function(V)
n=3
b=1-x[0]**2
#u.x.array[:]=0.9
#u.interpolate(lambda x:1-x[0]**2)
u.interpolate(lambda x: 2.0* (n + 2.0) ** (1.0 / n) * (1.0 - abs(x[0]) ** (1.0 + 1.0 / n)))
#b = fem.Constant(mesh, default_scalar_type(0.0))
alpha= ((n/(2.0*n+2.0))**n) / (n + 2.0)
Phi = -(2 * n + 2.0) / n * pow(u, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))

f= ufl.conditional(ufl.le(abs(x[0]), 0.5), 1, -1)
#f=1.0
functional = (alpha/(n+1)*(ufl.dot(ufl.grad(u)-Phi, ufl.grad(u)-Phi)**((n+1)/2.0))-f*u)*ufl.dx

dfunctional = ufl.derivative(functional, u, ufl.TestFunction(V))

ddfunctional = ufl.derivative(dfunctional, u, ufl.TrialFunction(V))

import sys
sys.path.append("../python/")
from snes_problem import SNESProblem

snes_problem = SNESProblem(dfunctional, ddfunctional, u, [bc])

b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J = dolfinx.fem.petsc.create_matrix(fem.form(snes_problem.a))

# Create Newton solver and solve
solver_snes = PETSc.SNES().create()
solver_snes.setType("vinewtonrsls")
solver_snes.setFunction(snes_problem.F, b)
solver_snes.setJacobian(snes_problem.J, J)
solver_snes.setTolerances(rtol=1.0e-5, max_it=50)
solver_snes.getKSP().setType("preonly")
solver_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_snes.getKSP().getPC().setType("lu")

# We set the bound (Note: they are passed as reference and not as values)
solver_snes.setVariableBounds(zero.vector,one.vector)

solver_snes.solve(None, u.vector)

print(u.x.array)

u_values = u.x.array
#u_values = u.x.array
x_values = np.linspace(start_point, end_point, num_intervals + 1)

if mesh.comm.rank == 0:
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(x_values, u_values, label="Numerical solution")
    plt.plot(x_values, 1 - x_values ** 2,
             label="u(x)",
             linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
   # plt.savefig("/home/zhenyu/SURE2024/7.10/test1.png")
    plt.show()
solver_snes.destroy()
