from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define mesh and function space
mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_e = Expression('0', degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_e, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Define the function for Dirichlet boundary condition
def u_D(x):
    return -0.5 * x**2 + 0.5 * x

# Generate x values for plotting
x_values = np.linspace(0, 1, 100)

# Evaluate the function at x values
u_values = u_D(x_values)

# Plot the function and FEniCS solution
plt.plot(x_values, u_values, label='exact solution', color='blue')
plt.plot(mesh.coordinates()[:, 0], u.compute_vertex_values(), label='FEniCS Solution', linestyle='dashed', color='red')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid(True)
plt.legend()
plt.show()
