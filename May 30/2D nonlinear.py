from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# Define mesh and function space
mesh = UnitSquareMesh(100,100)
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
b = Expression('-pow((x[0]-0.5), 2) + 0.25-pow((x[1]-0.5), 2)', degree=1)

#v=(u-b)^2 or (u-b)^3
#F = (u-b)**3*dot(grad(u), grad(v)) * dx - f * v * dx
F = (u-b)**2*dot(grad(u), grad(v)) * dx - f * v * dx
# Compute solution

solve(F == 0, u, bc)

# Plot the FEniCS solution
plot(mesh)
plot(u)
fig = plt.gcf()
ax = plt.gca()
cbar = plt.colorbar(plot(u), ax=ax)
plt.show()
