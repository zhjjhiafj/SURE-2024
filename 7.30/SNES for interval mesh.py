#Adapted from https://newfrac.gitlab.io/newfrac-fenicsx-training/03-variational-inequalities/VI.html
#Import required libraries
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

num_intervals = 1000
start_point = -1.0
end_point = 1.0
domain = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])


V = fem.functionspace(domain, ("CG", 1))

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(0)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(np.infty)
l = fem.Function(V)
with l.vector.localForm() as loc:
    loc.set(0)

def left(x):
    is_close = np.isclose(x[0], start_point)
    return is_close


def right(x):
    is_close = np.isclose(x[0], end_point )
    return is_close


left_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left)
left_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, left_facets)

right_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right)
right_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, right_facets)

bcs = [fem.dirichletbc(zero, left_dofs), fem.dirichletbc(zero, right_dofs)]

u = fem.Function(V)
u.interpolate(lambda x:1-x[0]**2)

ell = 0.3

def w(alpha):
    return alpha

functional = (ell * ufl.inner(ufl.grad(u), ufl.grad(u)) + w(u)/ell)*ufl.dx

dfunctional = ufl.derivative(functional, u, ufl.TestFunction(V))


ddfunctional = ufl.derivative(dfunctional, u, ufl.TrialFunction(V))

import sys
sys.path.append("../python/")
from snes_problem import SNESProblem

snes_problem = SNESProblem(dfunctional, ddfunctional, u, bcs)

b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J = dolfinx.fem.petsc.create_matrix(dolfinx.fem.form(snes_problem.a))

# Create Newton solver and solve
solver_snes = PETSc.SNES().create()
solver_snes.setType("vinewtonrsls")
solver_snes.setFunction(snes_problem.F, b)
solver_snes.setJacobian(snes_problem.J, J)
solver_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_snes.getKSP().setType("preonly")
solver_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_snes.getKSP().getPC().setType("lu")

# We set the bound (Note: they are passed as reference and not as values)
solver_snes.setVariableBounds(l.vector,one.vector)

solver_snes.solve(None, u.vector)

print(u.x.array)
u_values = u.x.array
#u_values = u.x.array
x_values = np.linspace(start_point, end_point, num_intervals + 1)

if domain.comm.rank == 0:
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(x_values, u_values, label="Numerical solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
   # plt.savefig("/home/zhenyu/SURE2024/7.10/test1.png")
    plt.show()
solver_snes.destroy()
