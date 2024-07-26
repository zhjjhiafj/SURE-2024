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
L = 1.0
H = 1.0

# Define the number of divisions in each direction
nx = 100
ny = 100

# Create the mesh
mesh = create_rectangle(MPI.COMM_WORLD,
                        [np.array([0, 0]), np.array([L, H])],
                        [nx, ny],
                        cell_type=dolfinx.mesh.CellType.quadrilateral)

# Write the mesh to a file
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as f:
    f.write_mesh(mesh)

x = ufl.SpatialCoordinate(mesh)

V = fem.functionspace(mesh, ("CG", 1))

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(np.infty)


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

tdim = mesh.topology.dim
fdim = tdim - 1
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(zero, boundary_dofs)

u=fem.Function(V)
u.x.array[:] = 0.9
n=3
b=1-0.6*ufl.sqrt(x[0]**2+x[1]**2)
from dolfinx import default_scalar_type
#b = fem.Constant(mesh, default_scalar_type(1.0))
alpha= ((n/(2.0*n+2.0))**n) / (n + 2.0)
Phi = -(2 * n + 2.0) / n * pow(u, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))
f=1
functional = (alpha/(n+1)*(ufl.dot(ufl.grad(u)-Phi,ufl.grad(u)-Phi)**((n+1)/2.0))-f*u)*ufl.dx

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
solver_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_snes.getKSP().setType("preonly")
solver_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_snes.getKSP().getPC().setType("lu")

# We set the bound (Note: they are passed as reference and not as values)
#solver_snes.setVariableBounds(zero.vector,one.vector)

solver_snes.solve(None, u.vector)

print(u.x.array)
from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/u.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(u)

topology, cell_types,geometry= dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
import pyvista
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["alpha"] = u.x.array.real
grid.set_active_scalars("alpha")
pyvista.OFF_SCREEN=False

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
figure = plotter.screenshot("output/u.png")
solver_snes.destroy()
