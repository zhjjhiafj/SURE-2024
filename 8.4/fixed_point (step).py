#penalty method that ensures the solution to be non-negative
import ufl
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
start_point = -1.0
end_point = 1.0
domain = mesh.create_interval(MPI.COMM_WORLD, num_intervals, [start_point, end_point])
x = ufl.SpatialCoordinate(domain)
V = fem.functionspace(domain, ("Lagrange", 1))

# u_ufl = (2**n*(n+2))**(1.0/(2*n+2))*(1-((x[0]**2)**1/2)**(1.0+1.0/n))**(n*1.0/(2*n+2))


# def u_exact(x):
#    return eval(str(u_ufl))


from dolfinx import default_scalar_type
num=1/((((n/(2.0*n+2.0))**n) / (n + 2.0)))
def f1(x):
    values = np.full_like(x[0], -3.0)
    mask = (x[0] > -0.5) & (x[0] < 0.5)
    values[mask] = 1.0
    return values
#f = fem.Constant(domain, default_scalar_type(num))
f=fem.Function(V)
f.interpolate(f1)

u_D = fem.Function(V)
u_D.interpolate(lambda x: np.zeros_like(x[0]))
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

error= float('inf')  # Use infinity as an initial error to ensure the loop starts
tolerance = 1e-9# Define a tolerance level
max_iterations =100  # Optional: to prevent infinite loops
iteration = 0
uh = fem.Function(V)
C=fem.Function(V)
constant_value = 1.0
C.interpolate(lambda x: np.full_like(x[0], constant_value))
b=1-x[0]**2
#b=fem.Constant(domain, default_scalar_type(0.0))
#b = 0.01* ufl.cos(1.2 * ufl.pi * abs(x[0]) )
epsilon=0.000000001
#b=1-x[0]**2

uh.interpolate(lambda x:(1-x[0]**2)*0.18)
while error > tolerance and iteration < max_iterations:
    uh2 = fem.Function(V)
    uh2.interpolate(lambda x:(1-x[0]**2)*0.18)
    #uh2.x.array[:]=0.04
    v = ufl.TestFunction(V)
    Phi = -(2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * (ufl.grad(b))
    a = (ufl.dot(ufl.grad(uh2) - Phi, ufl.grad(uh2) - Phi)) ** ((n - 1.0) / 2.0)
    # a = pow(ufl.dot(ufl.grad(uh2) - (2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * ufl.grad(b)\
    # , ufl.grad(uh2)
    #                - (2 * n + 2.0) / n * pow(uh, ((n + 2) / (2 * n + 2.0))) * ufl.grad(b)), (n - 1.0) / 2.0)
    k = ufl.div(a*Phi)
    penalty=uh2-(ufl.dot(uh2,uh2))**0.5*C
    #F = (a*ufl.dot(ufl.grad(uh2), ufl.grad(v))+k*v-(f)*v)*ufl.dx
    F = (a * ufl.dot(ufl.grad(uh2)-Phi, ufl.grad(v)) - (f-1/epsilon*penalty)* (v)) * ufl.dx
    #F = (a * ufl.dot(ufl.grad(uh2) - Phi, ufl.grad(v)) -f* v) * ufl.dx
    problem = NonlinearProblem(F, uh2, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criteron = "incremental"
    solver.rtol = 1e-7
    solver.report = True
    solver.max_it=2000
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "lu"
    #opts[f"{option_prefix}pc_type"] = "gamg"
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
    uh3=ufl.conditional(ufl.ge(uh2,0),uh2,0)
    uh4 = fem.Expression(uh3, V.element.interpolation_points())
    uh.interpolate(uh4)
    # Increment the iteration counter
    iteration += 1

# Compute L2 error and error at nodes
V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
# u_ex.interpolate(u_exact)
#u_values = uh.x.array**(n/(2*n+2.0))
f1=fem.Expression(f,V.element.interpolation_points())
f2=fem.Function(V)
f2.interpolate(f1)
u_values2=f2.x.array
u_values = uh.x.array
#u_values = abs(uh.x.array)**(n/(2*n+2.0))
x_values = np.linspace(start_point, end_point, num_intervals + 1)

if domain.comm.rank == 0:
    plt.show()
    plt.figure()
    plt.grid()
    plt.plot(x_values, u_values, label="Numerical solution")
    #plt.plot(x_values, u_values2, label="f")
    #plt.plot(x_values, 1-x_values**2,
     #        label="u(x)",
    #         linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.title("Penalty method a=1 (|x|<=0.5) a=-3 (0.5<|x|<=1)")
   # plt.savefig("/home/zhenyu/SURE2024/plt.plot(x_values, u_values, label="Numerical solution")7.10/test1.png")
    plt.show()

print(uh.x.array[20])
