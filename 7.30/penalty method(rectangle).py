#penalty method for rectangle mesh 
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
L = 1.0
H = 0.2

# Define the number of divisions in each direction
nx = 100
ny = 15

# Create the mesh
mesh = create_rectangle(MPI.COMM_WORLD,
                        [np.array([0.0, 0.0]), np.array([L, H])],
                        [nx, ny],
                        cell_type=dolfinx.mesh.CellType.quadrilateral)

# Write the mesh to a file
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as f:
    f.write_mesh(mesh)


V = fem.functionspace(mesh, ("CG", 1))

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(1.0)


def left(x):
    is_close = np.isclose(x[0], 0.0)
    return is_close


def right(x):
    is_close = np.isclose(x[0], L)
    return is_close

left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
left_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets)

right_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
right_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, right_facets)

bcs = [fem.dirichletbc(zero, left_dofs), fem.dirichletbc(one, right_dofs)]

u = fem.Function(V)

ell = 0.3

def w(alpha):
    return alpha

epsilon=0.000001

#penalty=1/epsilon/2*(ufl.inner(u,u)**0.5*(ufl.inner(u,u)**0.5-u))
penalty=1/epsilon/2*(u**2-abs(u)*u)
functional = (ell * ufl.inner(ufl.grad(u), ufl.grad(u)) + w(u)/ell+penalty)*ufl.dx

dfunctional = ufl.derivative(functional, u, ufl.TestFunction(V))

#ddfunctional = ufl.derivative(dfunctional, u, ufl.TrialFunction(V))
C=ufl.as_vector((1, 1))
problem = NonlinearProblem(dfunctional, u, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-5
solver.report = True
solver.max_it=2000
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "ilu"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)
m, converged = solver.solve(u)
assert (converged)
print(f"Number of interations: {m:d}")
