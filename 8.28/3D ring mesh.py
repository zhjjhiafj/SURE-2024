from mpi4py import MPI
import gmsh
from dolfinx.io import XDMFFile, gmshio

R = 1.0  # Outer radius of the ring
L = 0.75  # Inner radius of the ring


def gmsh_ring(model: gmsh.model, name: str) -> gmsh.model:
    """Create a Gmsh model of a ring-type geometry using 2D triangular cells."""
    model.add(name)
    model.setCurrent(name)

    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    # Create the outer and inner circles
    circle = model.occ.addDisk(0, 0, 0, R, R)
    circle_inner = model.occ.addDisk(0, 0, 0, L, L)

    # Cut the inner circle from the outer circle to form a ring
    cut = model.occ.cut([(2, circle)], [(2, circle_inner)])[0]

    # Synchronize the model
    model.occ.synchronize()

    # Add physical groups for the outer boundary, inner boundary, and the ring
    model.addPhysicalGroup(2, [cut[0][1]], tag=1)
    model.setPhysicalName(2, 1, "2D Ring")

    boundary_entities = model.getEntities(2)
    other_boundary_entities = []
    for entity in boundary_entities:
        if entity != cut[0][1]:
            other_boundary_entities.append(entity[1])
    model.addPhysicalGroup(2, other_boundary_entities, tag=3)
    model.setPhysicalName(2, 3, "Remaining boundaries")

    # Generate the 2D mesh
    model.mesh.generate(2)
    model.mesh.setOrder(2)

    return model


def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx from a Gmsh model and output to file."""
    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    with XDMFFile(msh.comm, filename, mode) as file:
        msh.topology.create_connectivity(2, 2)
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )


gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

# Create model
model = gmsh.model()
model = gmsh_ring(model, "Ring")
model.setCurrent("Ring")
create_mesh(MPI.COMM_SELF, model, "2d_ring",
            f"/home/zhenyu/SURE2024/8.28/out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w")

gmsh.finalize()
