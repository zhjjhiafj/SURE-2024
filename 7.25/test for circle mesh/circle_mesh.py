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
gmsh.initialize()
disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [disk], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
mesh.topology.create_connectivity(2,1)
x = ufl.SpatialCoordinate(mesh)

V = fem.functionspace(mesh, ("CG", 1))

zero = fem.Function(V)
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = fem.Function(V)
with one.vector.localForm() as loc:
    loc.set(np.infty)

def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)


boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)

bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)
u=fem.Function(V)
n=3
b=1-ufl.sqrt(x[0]**2+x[1]**2)
#u.x.array[:]=0.9
u.interpolate(lambda x:1-np.sqrt(x[0]**2+x[1]**2))
#b = fem.Constant(mesh, default_scalar_type(1.0))
alpha= ((n/(2.0*n+2.0))**n) / (n + 2.0)
Phi = -(2 * n + 2.0) / n * pow(u, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))

r = ufl.sqrt(x[0]**2 + x[1]**2)
f= ufl.conditional(ufl.le(r, 1/np.sqrt(2)), 1, -1)

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
solver_snes.setVariableBounds(zero.vector,one.vector)

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
