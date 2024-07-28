#radially symmetric problem edited from Jouvet and Bouler 5.1
import ufl
import dolfinx
import numpy
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
n=3
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


# Define the expression for b(x)
#b = b_0 * ufl.cos(z_0 * ufl.pi * abs(x[0]) / R)
from dolfinx import default_scalar_type
b=fem.Constant(domain, default_scalar_type(0.0))
u_0 = 1.0 - n /(n-1.0) * (
    (abs(x[0])/ R)**((n + 1.0) / (n ))
    - (1.0 - abs(x[0])/ R)**((n + 1.0) / (n ))
    + 1
    - (n + 1.0) / (n ) * (abs(x[0]) / R)
)
u=ufl.conditional(abs(x[0]) < R, u_0, 0)
h=u**(n/(2*n+2.0))
H=h + b
from dolfinx import default_scalar_type
q_plus=- ( 1/ (n + 2.0)) *H**(n + 2.0) * ufl.sqrt(ufl.dot(ufl.grad(h), ufl.grad(h)))\
                            **(n - 1.0) * ufl.grad(h)
q=ufl.conditional(abs(x[0]) < R, q_plus, ufl.as_vector([0.0] * len(q_plus)))
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
alpha=ufl.conditional(abs(x[0]) < R, alpha_plus, 0)
error = float('inf')  # Use infinity as an initial error to ensure the loop starts
tolerance = 1e-5  # Define a tolerance level
max_iterations = 100 # Optional: to prevent infinite loops
iteration = 0
uh = fem.Function(V)
u2=fem.Expression(u,V.element.interpolation_points())
uh.interpolate(lambda x:0.75**2-x[0]**2)
while error > tolerance and iteration < max_iterations:
    uh2 = fem.Function(V)
    uh2.interpolate(lambda x:1-x[0]**2)
    v = ufl.TestFunction(V)
    Phi = -(2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))
    a = ((n/(2.0*n+2.0))**n) / (n + 2.0)*(ufl.dot(ufl.grad(uh2) - Phi, ufl.grad(uh2) - Phi)) ** ((n - 1.0) / 2.0)
    # a = pow(ufl.dot(ufl.grad(uh2) - (2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * ufl.grad(b)\
    # , ufl.grad(uh2)
    #                - (2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * ufl.grad(b)), (n - 1.0) / 2.0)

    F = (a*ufl.dot(ufl.grad(uh2)-Phi, ufl.grad(v))-alpha*v)*ufl.dx

    problem = NonlinearProblem(F, uh2, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-5
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "ilu"
    #opts[f"{option_prefix}pc_type"] = "gamg"
    #opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.INFO)
    m, converged = solver.solve(uh2)
    assert (converged)
    print(f"Number of interations: {m:d}")

    L2_error = fem.form(ufl.inner(uh - uh2, uh - uh2) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    print(f"Iteration: {iteration}, Error: {error_L2}")
    uh=uh2
    # Increment the iteration counter
    iteration += 1

k=((abs(x[0])/R)**(1.0/n)+(1.0-abs(x[0])/R)**(1.0/n)-1.0)**n
# Compute L2 error and error at nodes
V_ex = fem.functionspace(domain, ("Lagrange", 1))
f1=fem.Expression(h,V_ex.element.interpolation_points())
f2=fem.Function(V_ex)
f2.interpolate(f1)
u_values=f2.x.array
u_values2 = uh.x.array**(n/(2*n+2.0))
x_values = np.linspace(start_point, end_point, num_intervals+1 )
b1=fem.Expression(b,V_ex.element.interpolation_points())
b2=fem.Function(V_ex)
b2.interpolate(b1)
b_values=b2.x.array

if domain.comm.rank == 0:
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(x_values, u_values2, label="Numerical solution")
    plt.plot(x_values, u_values, label="exact solution")
    plt.plot(x_values, b_values,
             label="b(x)",
             linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("Solution of the 1D Poisson equation")
   # plt.savefig("/home/zhenyu/SURE2024/7.10/test1.png")
    plt.show()

print(f2.x.array)
