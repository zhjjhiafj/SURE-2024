from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
mesh = UnitSquare(10,10)
# Create mesh and define function space
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_e= Expression('0', degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_e, boundary)
#u_D=Expression('1/2*pow(x[0],2)-1/2*x[0]', degree=2)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)
vtkfile = File('p3.pvd')
vtkfile << u

# Plot solution and mesh
plot(u)
plot(mesh)
#plot(u_D)
plt.show()
