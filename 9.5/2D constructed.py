import pyvista
from mpi4py import MPI
import ufl
import dolfinx
import math
import numpy
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
from dolfinx.io import XDMFFile, gmshio
import gmsh
# Initialize gmsh
gmsh.initialize()

# Create a new model named "filled_ring"
gmsh.model.add("filled_ring")

# Parameters
outer_radius = 1.0
inner_radius = 0.75
L=outer_radius
R=inner_radius
size=0.1
# Adjust mesh sizes for the points
outer_mesh_size = size  # Smaller value -> finer mesh
inner_mesh_size = size

import gmsh
import pyvista
from mpi4py import MPI
import dolfinx
from dolfinx.io import gmshio

# Initialize gmsh
gmsh.initialize()

# Create a new model named "ring_with_filled_inner_disk"
gmsh.model.add("ring_with_filled_inner_disk")


# Redefine points for the outer circle with the new mesh size
gmsh.model.geo.addPoint(outer_radius, 0, 0, outer_mesh_size, 1)
gmsh.model.geo.addPoint(0, outer_radius, 0, outer_mesh_size, 2)
gmsh.model.geo.addPoint(-outer_radius, 0, 0, outer_mesh_size, 3)
gmsh.model.geo.addPoint(0, -outer_radius, 0, outer_mesh_size, 4)
gmsh.model.geo.addPoint(0, 0, 0, outer_mesh_size, 5)  # Center point


# Define the outer circle
gmsh.model.geo.addCircleArc(1, 5, 2, 1)
gmsh.model.geo.addCircleArc(2, 5, 3, 2)
gmsh.model.geo.addCircleArc(3, 5, 4, 3)
gmsh.model.geo.addCircleArc(4, 5, 1, 4)

# Redefine points for the inner circle with the new mesh size
gmsh.model.geo.addPoint(inner_radius, 0, 0, inner_mesh_size, 6)
gmsh.model.geo.addPoint(0, inner_radius, 0, inner_mesh_size, 7)
gmsh.model.geo.addPoint(-inner_radius, 0, 0, inner_mesh_size, 8)
gmsh.model.geo.addPoint(0, -inner_radius, 0, inner_mesh_size, 9)

# Define the inner circle
gmsh.model.geo.addCircleArc(6, 5, 7, 5)
gmsh.model.geo.addCircleArc(7, 5, 8, 6)
gmsh.model.geo.addCircleArc(8, 5, 9, 7)
gmsh.model.geo.addCircleArc(9, 5, 6, 8)

# Create curve loops for the outer and inner circles
outer_loop = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
inner_loop = gmsh.model.geo.addCurveLoop([5, 6, 7, 8])

# Create plane surfaces for the inner disk and the ring
inner_surface = gmsh.model.geo.addPlaneSurface([inner_loop])
outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

# Synchronize the model
gmsh.model.geo.synchronize()

# Assign physical groups to the curves and surfaces
gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 1)  # Outer boundary
gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], 2)  # Inner boundary
gmsh.model.addPhysicalGroup(2, [outer_surface], 3)  # Outer ring surface
gmsh.model.addPhysicalGroup(2, [inner_surface], 4)  # Inner disk surface

# Synchronize the model again after adding physical groups
gmsh.model.geo.synchronize()

# Generate the mesh
gmsh.model.mesh.generate(2)

# Convert the Gmsh model to DOLFINx mesh
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)

# Finalize the gmsh session
gmsh.finalize()

#test for penalty method


n=3
x = ufl.SpatialCoordinate(domain)

V = fem.functionspace(domain, ("CG", 1))

b_0 = 0.01
z_0 = 1.2


# Define the expression for b(x)
#b=1-x[0]**2
b = b_0 * ufl.cos(z_0 * ufl.pi * abs(ufl.sqrt(x[0]**2+x[1]**2)) / R)
from dolfinx import default_scalar_type
#b=fem.Constant(domain, default_scalar_type(0.0))
u_0 = 1.0 - n /(n-1.0) * (
    (ufl.sqrt(x[0]**2+x[1]**2)/ R)**((n + 1.0) / (n ))
    - (1.0 - ufl.sqrt(x[0]**2+x[1]**2)/ R)**((n + 1.0) / (n ))
    + 1
    - (n + 1.0) / (n ) * (ufl.sqrt(x[0]**2+x[1]**2)/ R)
)
u=ufl.conditional(ufl.sqrt(x[0]**2+x[1]**2) < R, u_0, 0)
h=u**(n/(2*n+2.0))
H=h + b
from dolfinx import default_scalar_type
q_plus=- ( 1/ (n + 2.0)) *H**(n + 2.0) * ufl.sqrt(ufl.dot(ufl.grad(h), ufl.grad(h)))\
                            **(n - 1.0) * ufl.grad(h)
q=ufl.conditional(ufl.sqrt(x[0]**2+x[1]**2) < R, q_plus, ufl.as_vector([0.0] * len(q_plus)))
f=ufl.div(q)
u_D = fem.Function(V)
u_D.interpolate(lambda x: np.zeros_like(x[0]))
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))
Phi_ufl = -(2 * n + 2.0) / n * pow(u, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))
alpha_plus=(-((n/(2.0*n+2.0))**n) / (n + 2.0)*
       ufl.div((ufl.dot(ufl.grad(u) - Phi_ufl, ufl.grad(u) - Phi_ufl)) ** ((n - 1.0) / 2.0)\
           * (ufl.grad(u) - Phi_ufl)))
alpha2=fem.Function(V)
alpha3=fem.Expression(alpha_plus,V.element.interpolation_points())
alpha2.interpolate(alpha3)
alpha=ufl.conditional(ufl.sqrt(x[0]**2+x[1]**2) < R-1e-6, alpha_plus, -2.46)
alpha5=fem.Expression(alpha,V.element.interpolation_points())
alpha4=fem.Function(V)
alpha4.interpolate(alpha5)
#alpha=ufl.conditional(abs(x[0]) < R, alpha_plus, 0)
error = float('inf')  # Use infinity as an initial error to ensure the loop starts
tolerance = 1e-15# Define a tolerance level
max_iterations = 6 # Optional: to prevent infinite loops
iteration = 0
array= np.arange(max_iterations*1.0)
uh = fem.Function(V)
u2=fem.Expression(u,V.element.interpolation_points())
#uh.interpolate(lambda x:(1-(x[0]**2+x[1]**2))*0.8)
uh.x.array[:]=1.0
C=fem.Function(V)
constant_value = 1.0
C.interpolate(lambda x: np.full_like(x[0], constant_value))
epsilon=1e-8
while iteration < max_iterations and error > tolerance:
    uh2 = fem.Function(V)
    uh2.interpolate(lambda x:((1-(x[0]**2+x[1]**2)))*0.8)
    #uh2.x.array[:] = 0.5
    v = ufl.TestFunction(V)
    Phi = -(2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))
    a = ((n/(2.0*n+2.0))**n) / (n + 2.0)*(ufl.dot(ufl.grad(uh2) - Phi, ufl.grad(uh2) - Phi)) ** ((n - 1.0) / 2.0)
    # a = pow(ufl.dot(ufl.grad(uh2) - (2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * ufl.grad(b)\
    # , ufl.grad(uh2)
    #                - (2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * ufl.grad(b)), (n - 1.0) / 2.0)
    penalty = uh2 - (ufl.dot(uh2, uh2)) ** 0.5 * C

    F = (a*ufl.dot(ufl.grad(uh2)-Phi, ufl.grad(v))-(alpha-1/epsilon*penalty)*v)*ufl.dx

    problem = NonlinearProblem(F, uh2, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "residual"
    #solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol=1e-10
    ksp = solver.krylov_solver
    log.set_log_level(log.LogLevel.INFO)
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    #opts[f"{option_prefix}pc_type"] = "ilu"
    solver.max_it = 2000

    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.INFO)
    m, converged = solver.solve(uh2)
    assert (converged)
    print(f"Number of interations: {m:d}")

    L2_error = fem.form(ufl.inner(uh - uh2, uh - uh2) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    error=error_L2
    print(f"Iteration: {iteration}, Error: {error_L2}")
    array[iteration] = np.log10(error_L2)
    # Increment the iteration counter
    uh3 = ufl.conditional(ufl.ge(uh2, 0), uh2, 0)
    uh4 = fem.Expression(uh3, V.element.interpolation_points())
    uh.interpolate(uh4)
    iteration += 1

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/u.xdmf", "w") as f:
    f.write_mesh(domain)
    f.write_function(uh)

u2=fem.Expression(u,V.element.interpolation_points())
u3=fem.Function(V)
u3.interpolate(u2)
error_local = fem.assemble_scalar(fem.form((uh - u3)**2 * ufl.dx))
error_L2 = numpy.log(numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM)))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = numpy.log(domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u3.x.array)), op=MPI.MAX))
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
topology, cell_types,geometry= dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
import pyvista
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["alpha"] = uh.x.array.real-u3.x.array.real
#grid.point_data["alpha"] = alpha4.x.array.real
#grid.point_data["alpha"] = uh.x.array.real
grid.set_active_scalars("alpha")
pyvista.OFF_SCREEN=False
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
figure = plotter.screenshot("output/u.png")
