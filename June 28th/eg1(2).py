import dolfinx.fem.petsc
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, io, nls, log
import numpy as np
import pyvista
import ufl
from mpi4py import MPI
from petsc4py import PETSc
n=1.1
def q(u):
    #return 1.0/3.0*u**3
    return (1.0 / (n + 2.0)) * u ** (n + 2.0) * ((ufl.dot(ufl.grad(u), ufl.grad(u))) ** 2) ** ((n - 1.0) / 2.0)

num_intervals = 500
start_point = -1.0
end_point = 1.0
domain = mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])
from dolfinx import default_scalar_type
f=fem.Constant(domain, default_scalar_type(1))
x = ufl.SpatialCoordinate(domain)
#u_ufl = 6**0.25*(1.0-x[0]**2.0)**0.25
#u_ufl=6.0**0.25*(1.0-((x[0]**2.0)**0.5)**2.0)**0.25
u_ufl = (2**n*(n+2))**(1.0/(2*n+2))*(1-((x[0]**2)**1/2)**(1.0+1.0/n))**(n*1.0/(2*n+2))
def u_exact(x):
    return eval(str(u_ufl))


V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
u_D = dolfinx.fem.Function(V)
u_D.interpolate(lambda x: np.zeros_like(x[0]))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, fdim + 1)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))
uh = dolfinx.fem.Function(V)
uh.interpolate(lambda x: 1.6-1.6*x[0]**2)
#uh.interpolate(lambda x: (((2**n)*(n+2)*1.0)**(1.0/(2*n+2)))*(1-((x[0]**2)**0.5)**(1.0+1.0/n))**(n*1.0/(2*n+2)))
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
J = ufl.derivative(F, uh)
residual = dolfinx.fem.form(F)
jacobian = dolfinx.fem.form(J)
du = dolfinx.fem.Function(V)
A = dolfinx.fem.petsc.create_matrix(jacobian)
L = dolfinx.fem.petsc.create_vector(residual)
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
i = 0
error = dolfinx.fem.form(ufl.inner(uh - u_ufl, uh - u_ufl) * ufl.dx(metadata={"quadrature_degree": 4}))
L2_error = []
du_norm = []
max_iterations = 50
while i < max_iterations:
    # Assemble Jacobian and residual
    with L.localForm() as loc_L:
        loc_L.set(0)
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, jacobian, bcs=[bc])
    A.assemble()
    dolfinx.fem.petsc.assemble_vector(L, residual)
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    L.scale(-1)

    # Compute b - J(u_D-u_(i-1))
    dolfinx.fem.petsc.apply_lifting(L, [jacobian], [[bc]], x0=[uh.vector], scale=1)
    # Set du|_bc = u_{i-1}-u_D
    dolfinx.fem.petsc.set_bc(L, [bc], uh.vector, 1.0)
    L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    solver.solve(L, du.vector)
    du.x.scatter_forward()

    # Update u_{i+1} = u_i + delta u_i
    uh.x.array[:] += du.x.array
    i += 1

    # Compute norm of update
    correction_norm = du.vector.norm(0)

    # Compute L2 error comparing to the analytical solution
    L2_error.append(np.sqrt(MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)))
    du_norm.append(correction_norm)

    print(f"Iteration {i}: Correction norm {correction_norm}, L2 error: {L2_error[-1]}")
    if correction_norm < 1e-12:
        break

fig = plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.plot(np.arange(i), L2_error)
plt.title(r"$L^2(\Omega)$-error of $u_h$")
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel("Iterations")
plt.ylabel(r"$L^2$-error")
plt.grid()
plt.subplot(122)
plt.title(r"Residual of $\vert\vert\delta u_i\vert\vert$")
plt.plot(np.arange(i), du_norm)
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel("Iterations")
plt.ylabel(r"$\vert\vert \delta u\vert\vert$")
plt.grid()
import matplotlib.pyplot as plt
plt.show()

error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
u_values = uh.x.array
x_values = np.linspace(start_point, end_point, num_intervals + 1)
if domain.comm.rank == 0:
    plt.figure()
    plt.plot(x_values, u_values, label="Numerical solution")
    #plt.plot(x_values, 6.0**0.25*(1.0-((x_values**2.0)**0.5)**2.0)**0.25, label="Exact solution", linestyle='dashed')
    plt.plot(x_values, ((2.0**n*(n+2.0))**(1.0/(2.0*n+2.0)))*(1.0-((x_values**2)**0.5)**(1.0+1.0/n))**(n*1.0/(2.0*n+2.0)), label="Exact solution",
             linestyle='dashed')

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
   # plt.savefig("/home/zhenyu/SURE2024/6.27/test2.png")
    plt.show()
