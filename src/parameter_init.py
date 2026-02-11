
from dolfinx import fem
from petsc4py import PETSc
import numpy as np

def par_cytoplasm_init(msh):
    E = fem.Constant(msh, PETSc.ScalarType(1.0))
    nu = fem.Constant(msh, PETSc.ScalarType(0.3))
    mu1 = fem.Constant(msh, PETSc.ScalarType(0.1))
    mu2 = fem.Constant(msh, PETSc.ScalarType(0.05))
    c_coupling = fem.Constant(msh, PETSc.ScalarType(1.0))

    # zeta = fem.Constant(msh, PETSc.ScalarType(5.0))  # drag in nucleus
    # eta_f = fem.Constant(msh, PETSc.ScalarType(0.1))  # fluid viscosity 
    # lam_s = fem.Constant(msh, PETSc.ScalarType(1.0))  # solid Lamé 
    # mu_s = fem.Constant(msh, PETSc.ScalarType(0.5))   # solid shear 

    # _dt = fem.Constant(msh, PETSc.ScalarType(1e-2))

    return E, nu, mu1, mu2, c_coupling

def par_nucleus_init_viscoelastic(msh):
    E = fem.Constant(msh, PETSc.ScalarType(0.2))
    nu = fem.Constant(msh, PETSc.ScalarType(0.1))
    mu1 = fem.Constant(msh, PETSc.ScalarType(0.1))
    mu2 = fem.Constant(msh, PETSc.ScalarType(0.05))
    c_coupling = fem.Constant(msh, PETSc.ScalarType(1.0))

    # zeta = fem.Constant(msh, PETSc.ScalarType(5.0))  # drag in nucleus
    # eta_f = fem.Constant(msh, PETSc.ScalarType(0.1))  # fluid viscosity 
    # lam_s = fem.Constant(msh, PETSc.ScalarType(1.0))  # solid Lamé 
    # mu_s = fem.Constant(msh, PETSc.ScalarType(0.5))   # solid shear 

    # _dt = fem.Constant(msh, PETSc.ScalarType(1e-2))

    return E, nu, mu1, mu2, c_coupling


def init_state(V, Q):
    gdim = V.mesh.geometry.dim

    u = fem.Function(V, name="u")
    u_prev = fem.Function(V, name="u_prev")

    p = fem.Function(Q, name="p")
    p_prev = fem.Function(Q, name="p_prev")


    # u(t=0) = 0, 0, 0

    def u_init(x):
        return np.vstack((0.0*x[0], 0.0*x[1], 0.0*x[2]))
    
    u.interpolate(u_init)
    u_prev.interpolate(u_init)

    # p(t=0) = z
    def p_init(x):
        return 0.01*x[2]

    p.interpolate(p_init)
    p_prev.interpolate(p_init)


    u.x.scatter_forward()
    u_prev.x.scatter_forward()
    p.x.scatter_forward()
    p_prev.x.scatter_forward()

    return u, u_prev, p, p_prev




def init_state_nucleus(V, Q):
    gdim = V.mesh.geometry.dim

    u = fem.Function(V, name="u_inner")
    u_prev = fem.Function(V, name="u_inner_prev")

    p = fem.Function(Q, name="p")
    p_prev = fem.Function(Q, name="p_prev")


    # u(t=0) = 0, 0, 0

    def u_init(x):
        return np.vstack((0.0*x[0], 0.0*x[1], 0.0*x[2]))
    
    u.interpolate(u_init)
    u_prev.interpolate(u_init)

    # p(t=0) = y
    def p_init(x):
        return x[1]

    p.interpolate(p_init)
    p_prev.interpolate(p_init)


    u.x.scatter_forward()
    u_prev.x.scatter_forward()
    p.x.scatter_forward()
    p_prev.x.scatter_forward()

    return u, u_prev, p, p_prev


def boundary_state(V, Q):
    gdim = V.mesh.geometry.dim

    u = fem.Function(V, name="u")
    u_prev = fem.Function(V, name="u_prev")

    p = fem.Function(Q, name="p")
    p_prev = fem.Function(Q, name="p_prev")


    # u(t=0) = 0, 0, 0

    def u_init(x):
        return np.vstack((0.0*x[0], 0.0*x[1], 0.0*x[2]))
    
    u.interpolate(u_init)
    u_prev.interpolate(u_init)

    # p(t=0) = z
    def p_init(x):
        return x[2]

    p.interpolate(p_init)
    p_prev.interpolate(p_init)


    u.x.scatter_forward()
    u_prev.x.scatter_forward()
    p.x.scatter_forward()
    p_prev.x.scatter_forward()

    return u, u_prev, p, p_prev

