from __future__ import annotations

from dolfinx import fem
import numpy as np
import ufl


def is_boundary_close(x, R, atol=1e-6):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return np.isclose(r, R, atol=atol)


def _zero_scalar_function(Q):
    p_bc = fem.Function(Q)
    p_bc.x.array[:] = 0.0
    p_bc.x.scatter_forward()
    return p_bc


# -------------------------
# Dirichlet for pressure p
# -------------------------

def set_bc_outer_dirichlet(R_outer, Q, *, value=None, atol=1e-6):
    """
    p = value
    Returns: (bc, F_bc)
    """
    dofs = fem.locate_dofs_geometrical(Q, lambda x: is_boundary_close(x, R_outer, atol))
    if len(dofs) == 0:
        raise RuntimeError(f"No outer boundary points")

    if value is None:
        value = _zero_scalar_function(Q)

    bc = fem.dirichletbc(value, dofs)
    return bc, 0


def set_bc_inner_dirichlet(R_inner, Q, *, value=None, atol=1e-6):
    """
    p = value 
    Returns: (bc, F_bc)
    """
    dofs = fem.locate_dofs_geometrical(Q, lambda x: is_boundary_close(x, R_inner, atol))
    if len(dofs) == 0:
        raise RuntimeError(f"No inner boundary points")

    if value is None:
        value = _zero_scalar_function(Q)

    bc = fem.dirichletbc(value, dofs)
    return bc, 0


# -------------------------
# Neumann for pressure p
# -------------------------
# Weak term: +∫ g * q ds

def set_bc_outer_neumann_flux(g, q, ds_outer):
    """
    (K grad p)·n = g on outer boundary
    Returns: (bc, F_bc)
    """
    bc = None
    F_bc = g * q * ds_outer
    return bc, F_bc


def set_bc_inner_neumann_flux(g, q, ds_inner):
    """
    (K grad p)·n = g on inner boundary
    Returns: (bc, F_bc)
    """
    bc = None
    F_bc = g * q * ds_inner
    return bc, F_bc


# -------------------------
# Robin prototypes
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
