from __future__ import annotations

from dolfinx import fem
from petsc4py import PETSc
import numpy as np
import ufl


def is_boundary_close(x, R, atol=1e-6):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    return np.isclose(r, R, atol=atol)


def _zero_vec_function(V):
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    u_bc.x.scatter_forward()
    return u_bc


# -------------------------
# OUTER boundary
# -------------------------

def set_bc_outer_dirichlet(R_outer, V_outer, *, value=None, atol=1e-6):
    """
    Returns: (bc, F_bc)
    Dirichlet u = value on |x|=R_outer.
    """
    dofs_outer = fem.locate_dofs_geometrical(V_outer, lambda x: is_boundary_close(x, R_outer, atol))
    if len(dofs_outer) == 0:
        raise RuntimeError(f"No outer boundary points")

    if value is None:
        value = _zero_vec_function(V_outer)

    bc = fem.dirichletbc(value, dofs_outer)
    F_bc = 0
    return bc, F_bc


def set_bc_outer_neumann(t, v, ds_outer):
    """
    Returns: (bc, F_bc)
    Neumann traction: Sigma n = t on outer boundary.
    """
    bc = None
    F_bc = -ufl.dot(t, v) * ds_outer
    return bc, F_bc


# -------------------------
# INNER boundary
# -------------------------

def set_bc_inner_dirichlet(R_inner, V_outer, *, value=None, atol=1e-6):
    """
    Returns: (bc, F_bc)
    Dirichlet u = value on inner boundary
    """
    dofs_inner = fem.locate_dofs_geometrical(V_outer, lambda x: is_boundary_close(x, R_inner, atol))
    if len(dofs_inner) == 0:
        raise RuntimeError(f"No inner boundary points")

    if value is None:
        value = _zero_vec_function(V_outer)

    bc = fem.dirichletbc(value, dofs_inner)
    F_bc = 0
    return bc, F_bc


def set_bc_inner_neumann(t, v, ds_inner):
    """
    Returns: (bc, F_bc)
    Neumann traction: Sigma n = t on inner boundary.
    """
    bc = None
    F_bc = -ufl.dot(t, v) * ds_inner
    return bc, F_bc




# -------------------------
# ROBIN prototypes
# -------------------------

def robin_outer():
    """
    TODO:
    Returns: (bc, F_bc)
    """
    return None, 0


def robin_inner():
    """
    TODO:
    Returns: (bc, F_bc)
    """
    return None, 0
