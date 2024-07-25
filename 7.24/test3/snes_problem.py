from typing import List
from dolfinx.la import create_petsc_vector
from petsc4py import PETSc
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)
import dolfinx
import ufl

from dolfinx import fem
class SNESProblem:
    """Nonlinear problem class compatible with PETSC.SNES solver.
    """

    def __init__(self, F: ufl.form.Form, J: ufl.form.Form, u: fem.Function, bcs: List[fem.dirichletbc]):
        """This class set up structures for solving a non-linear problem using Newton's method.

        Parameters
        ==========
        F: Residual.
        J: Jacobian.
        u: Solution.
        bcs: Dirichlet boundary conditions.
        """
        self.L = F
        self.a = J
        self.bcs = bcs
        self.u = u

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.A = dolfinx.fem.petsc.create_matrix(dolfinx.fem.form(self.a))
        self.b = dolfinx.fem.petsc.create_vector(dolfinx.fem.form(self.L))

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # We need to assign the vector to the function

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Zero the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, dolfinx.fem.form(self.L))

        # Apply boundary conditions
        dolfinx.fem.apply_lifting(b, [dolfinx.fem.form(self.a)], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        assemble_matrix(A, dolfinx.fem.form(self.a), bcs=self.bcs)
        A.assemble()
