#snes solver for bound constrained minimization problem
#Adapted from https://newfrac.gitlab.io/newfrac-fenicsx-training/03-variational-inequalities/VI.html
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
import math
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle
n=2
num_intervals = 1000
L=1.0
R=0.75
start_point = -L
end_point = L
domain = mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])
x = ufl.SpatialCoordinate(domain)
V = fem.functionspace(domain, ("Lagrange", 1))

b_0 = 0.01
z_0 = 1.2

# Write the mesh to a file
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as f:
    f.write_mesh(domain)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as f:
    f.write_mesh(domain)

V = fem.functionspace(domain, ("Lagrange", 1))
v = ufl.TestFunction(V)
# Define the expression for b(x)
boundary = b_0 * ufl.cos(z_0 * ufl.pi * abs(x[0]) / R)

u_0 = 1.0 - n /(n-1.0) * (
    (abs(x[0])/ R)**((n + 1.0) / (n ))
    - (1.0 - abs(x[0])/ R)**((n + 1.0) / (n ))
    + 1
    - (n + 1.0) / (n ) * (abs(x[0]) / R)
)
u=ufl.conditional(abs(x[0]) < R, u_0, 0)
h=u**(n/(2*n+2.0))
H=h + boundary
q_plus=- ( 1/ (n + 2.0)) *H**(n + 2.0) * ufl.sqrt(ufl.dot(ufl.grad(h), ufl.grad(h)))\
                            **(n - 1.0) * ufl.grad(h)
q=ufl.conditional(abs(x[0]) < R, q_plus, ufl.as_vector([0.0] * len(q_plus)))
f=ufl.div(q)


Phi_ufl = -(2 * n + 2.0) / n * pow(u, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(boundary))
alpha_plus=(-((n/(2.0*n+2.0))**n) / (n + 2.0)*
       ufl.div((ufl.dot(ufl.grad(u) - Phi_ufl, ufl.grad(u) - Phi_ufl)) ** ((n - 1.0) / 2.0)\
           * (ufl.grad(u) - Phi_ufl)))
alpha2=fem.Function(V)
alpha3=fem.Expression(alpha_plus,V.element.interpolation_points())
alpha2.interpolate(alpha3)
alpha=ufl.conditional(abs(x[0]) < R, alpha_plus, alpha2.x.array[math.ceil(((L-R)/L/2.0)*num_intervals)+1])
uh = fem.Function(V)
Phi = -(2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(boundary))
A = ((n/(2.0*n+2.0))**n) / (n + 2.0)*(ufl.dot(ufl.grad(uh) - Phi, ufl.grad(uh) - Phi)) ** ((n - 1.0) / 2.0)
dfunctional =(A*ufl.dot(ufl.grad(uh)-Phi, ufl.grad(v))-alpha*v)*ufl.dx

#ddfunctional = ufl.derivative(dfunctional, u, ufl.TrialFunction(V))

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(-0.01)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(np.infty)

u_D = fem.Function(V)
u_D.interpolate(lambda x: np.zeros_like(x[0]))
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

import sys
sys.path.append("../python/")
from snes_problem import SNESProblem
# Create nonlinear problem
problem = SNESProblem(dfunctional, uh, [bc])
u2=fem.Expression((1-x[0]**2),V.element.interpolation_points())
uh.interpolate(u2)
#uh.x.array[:] = 0.7
b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J = dolfinx.fem.petsc.create_matrix(problem.a)

# Create Newton solver and solve
snes = PETSc.SNES().create()
snes.setType("vinewtonrsls")
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-9, max_it=100)
snes.getKSP().setType("preonly")
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.getKSP().getPC().setType("ilu")
#snes.setVariableBounds(zero.vector,one.vector)
snes.solve(None, uh.x.petsc_vec)

print(snes.getIterationNumber())
print(snes.getFunctionNorm())
print(uh.x.array)

V_ex = fem.functionspace(domain, ("Lagrange", 1))
f1=fem.Expression(u,V_ex.element.interpolation_points())
f2=fem.Function(V_ex)
f2.interpolate(f1)
u_values=f2.x.array
u_values2 = uh.x.array
x_values = np.linspace(start_point, end_point, num_intervals+1 )


if domain.comm.rank == 0:
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(x_values, u_values2, label="Numerical solution")
    plt.plot(x_values, u_values, label="exact solution")
    #plt.plot(x_values, b_values,label="b(x)",linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
   # plt.savefig("/home/zhenyu/SURE2024/7.10/test1.png")
    plt.show()

snes.destroy()
b.destroy()
J.destroy()
