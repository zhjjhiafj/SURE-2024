#SNES solver for bound_constrained problem 
#Adpated from dolfinx test problem https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/nls/test_newton.py#L179-L202

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
mesh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 15)
V = fem.functionspace(mesh, ("Lagrange", 1))
u = fem.Function(V)
v = ufl.TestFunction(V)
F = ufl.inner(5.0, v) * ufl.dx - ufl.sqrt(u * u) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(u, v) * ufl.dx

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(0.8)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(1.3)
u_bc = fem.Function(V)
u_bc.x.array[:] = 1.0
bc = fem.dirichletbc(
    u_bc,
    fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)),
)
import sys
sys.path.append("../python/")
from snes_problem import SNESProblem
# Create nonlinear problem
problem = SNESProblem(F, u, bc)

u.x.array[:] = 0.9
b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J = create_matrix(problem.a)

# Create Newton solver and solve
snes = PETSc.SNES().create()
snes.setType("vinewtonrsls")
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-9, max_it=10)
snes.getKSP().setType("preonly")
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.getKSP().getPC().setType("lu")
snes.setVariableBounds(zero.vector,one.vector)
snes.solve(None, u.x.petsc_vec)

# print(snes.getIterationNumber())
# print(snes.getFunctionNorm())



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
figure = plotter.screenshot("output/u.png")

snes.destroy()
b.destroy()
J.destroy()
