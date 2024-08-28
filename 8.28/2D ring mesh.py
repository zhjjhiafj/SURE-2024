import gmsh
import pyvista
from mpi4py import MPI
import dolfinx
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags

# Initialize gmsh
gmsh.initialize()

# Create a new model named "ring"
gmsh.model.add("ring")

# Parameters
outer_radius = 1.0
inner_radius = 0.5

# Define points for the outer circle
gmsh.model.geo.addPoint(outer_radius, 0, 0, 0.1, 1)
gmsh.model.geo.addPoint(0, outer_radius, 0, 0.1, 2)
gmsh.model.geo.addPoint(-outer_radius, 0, 0, 0.1, 3)
gmsh.model.geo.addPoint(0, -outer_radius, 0, 0.1, 4)
gmsh.model.geo.addPoint(0, 0, 0, 0.1, 5)  # Center point

# Define the outer circle
gmsh.model.geo.addCircleArc(1, 5, 2, 1)
gmsh.model.geo.addCircleArc(2, 5, 3, 2)
gmsh.model.geo.addCircleArc(3, 5, 4, 3)
gmsh.model.geo.addCircleArc(4, 5, 1, 4)

# Define points for the inner circle
gmsh.model.geo.addPoint(inner_radius, 0, 0, 0.1, 6)
gmsh.model.geo.addPoint(0, inner_radius, 0, 0.1, 7)
gmsh.model.geo.addPoint(-inner_radius, 0, 0, 0.1, 8)
gmsh.model.geo.addPoint(0, -inner_radius, 0, 0.1, 9)

# Define the inner circle
gmsh.model.geo.addCircleArc(6, 5, 7, 5)
gmsh.model.geo.addCircleArc(7, 5, 8, 6)
gmsh.model.geo.addCircleArc(8, 5, 9, 7)
gmsh.model.geo.addCircleArc(9, 5, 6, 8)

# Create curve loops for the outer and inner circles
outer_loop = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
inner_loop = gmsh.model.geo.addCurveLoop([5, 6, 7, 8])

# Create a plane surface with the outer loop and subtract the inner loop
surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

# Synchronize the model
gmsh.model.geo.synchronize()

# Assign physical groups to the curves and surface
gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 1)  # Outer boundary
gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], 2)  # Inner boundary
gmsh.model.addPhysicalGroup(2, [surface], 3)     # Surface

# Synchronize the model
gmsh.model.geo.synchronize()
# Generate the mesh
gmsh.model.mesh.generate(2)
# Convert the Gmsh model to DOLFINx mesh
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)

domain.topology.create_connectivity(2, 1)
gmsh.finalize()
# Finalize the gmsh session

topology, cell_types,geometry= dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
# Visualize the mesh using PyVista
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
pyvista.OFF_SCREEN=False
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
plotter.view_xy()
plotter.show()
