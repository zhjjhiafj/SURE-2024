# The plot of example 1 with n=1, L=1, a_0=1 in Bueler's document
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define mesh and function space
mesh = IntervalMesh(100,-1,1)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_e = Expression('0', degree=1)

def boundary(x, on_boundary):
    return on_boundary

# bc = DirichletBC(V, u_e, boundary)
bc = DirichletBC(V, u_e, boundary)

# Define variational problem
u = Function(V)
v = TestFunction(V)
f=Constant(1)
# Define expression for b
u.interpolate(Constant(0.5))
F = 1/3*u**3*dot(grad(u), grad(v)) * dx - f * v * dx
# Compute solution
solve(F == 0, u, bc)
# Define the exact solution b for plotting 
def b(x):
    return 6**0.25*(1-x**2)**0.25
# Generate x values for plotting
x_values = np.linspace(-1, 1, 100)

# Evaluate the function at x values
b_values = np.array([b(x) for x in x_values])

# Plot b
plt.plot(x_values, b_values, label='Exact Solution', color='blue')

# Plot the FEniCS solution
plt.plot(mesh.coordinates()[:, 0], u.compute_vertex_values(), label='FEniCS Solution',linestyle='dashed', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
