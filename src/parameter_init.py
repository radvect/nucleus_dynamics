
from ufl import Identity
from dolfinx import fem
from petsc4py import PETSc

def par_init(msh):
    I = Identity(msh.topology.dim)
    E = fem.Constant(msh, PETSc.ScalarType(1.0))
    nu = fem.Constant(msh, PETSc.ScalarType(0.3))
    mu1 = fem.Constant(msh, PETSc.ScalarType(0.1))
    mu2 = fem.Constant(msh, PETSc.ScalarType(0.05))
    c_coupling = fem.Constant(msh, PETSc.ScalarType(1.0))

    zeta = fem.Constant(msh, PETSc.ScalarType(5.0))  # drag in nucleus
    eta_f = fem.Constant(msh, PETSc.ScalarType(0.1))  # fluid viscosity (Brinkman)
    lam_s = fem.Constant(msh, PETSc.ScalarType(1.0))  # solid Lamé (placeholder)
    mu_s = fem.Constant(msh, PETSc.ScalarType(0.5))   # solid shear (placeholder)

    # Time step
    _dt = fem.Constant(msh, PETSc.ScalarType(1e-2))

    # Useful elastic/visc coefficients for cytoplasm (small-strain Kelvin-Voigt + coupling c*P*I)
    nu_prime = nu / (1.0 - 2.0 * nu)

    return I, E, nu, mu1, mu2, c_coupling, zeta, eta_f, lam_s, mu_s, _dt, nu_prime