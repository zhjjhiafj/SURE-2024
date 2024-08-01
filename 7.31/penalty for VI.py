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
x = ufl.SpatialCoordinate(mesh)
u = fem.Function(V)
u.x.array[:] = 0.3
ell = 0.3

def w(alpha):
    return alpha

epsilon=0.00000000001

#penalty=1/epsilon/2*(ufl.inner(u,u)**0.5*(ufl.inner(u,u)**0.5-u))
#penalty=1/epsilon/2*(u**2-abs(u)*u)
penalty=1/(2*epsilon)*ufl.conditional(-u>0,-u,0)**2
functional = (ell * ufl.inner(ufl.grad(u), ufl.grad(u)) + w(u)/ell+penalty)*ufl.dx

dfunctional = ufl.derivative(functional, u, ufl.TestFunction(V))

problem = NonlinearProblem(dfunctional, u, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-10
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

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/u.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(u)

import dolfinx.plot
topology, cell_types,geometry= dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
import pyvista
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["alpha"] = u.x.array.real
grid.set_active_scalars("alpha")

if pyvista.OFF_SCREEN:
    from pyvista.utilities.xvfb import start_xvfb
    start_xvfb(wait=0.1)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
#figure = plotter.screenshot("output/u.png")
print(u.x.array)
