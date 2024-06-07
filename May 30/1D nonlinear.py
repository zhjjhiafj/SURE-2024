from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define mesh and function space
mesh = UnitIntervalMesh(100)
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
#f = Expression('-pow((x[0]-0.5), 2) + 0.125', degree=1)
f=Constant(1)
# Define expression for b
b = Expression('-1/2*pow((x[0]-0.5), 2) + 0.125', degree=1)

#v=(u-b)^2 or (u-b)^3
F = (u-b)**3*dot(grad(u), grad(v)) * dx - f * v * dx
#F = (u-b)**2*dot(grad(u), grad(v)) * dx - f * v * dx
# Compute solution

solve(F == 0, u, bc)
# Define function b for plotting
def b(x):
    return -1/2*(x-0.5)**2+0.125

# Generate x values for plotting
x_values = np.linspace(0, 1, 100)

# Evaluate the function at x values
b_values = b(x_values)

# Plot b
plt.plot(x_values, b_values, label='b', color='blue')

# Plot the FEniCS solution
plt.plot(mesh.coordinates()[:, 0], u.compute_vertex_values(), label='h', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
