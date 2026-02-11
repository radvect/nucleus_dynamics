from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
from dolfinx import fem, io, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from src.mesh_initialization import build_two_region_ball_mesh
from src.parameter_init import par_cytoplasm_init, par_nucleus_init_viscoelastic, init_state_nucleus, init_state
import src.boundary_setup_displacement as bc_u


msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()
tdim = msh.topology.dim
gdim = msh.geometry.dim

### commented cytoplasm initializing isn't needed for this test.
# E, nu, mu1, mu2, c_coupling = par_cytoplasm_init(msh)
# I = ufl.Identity(gdim)

### commented cytoplasm initializing isn't needed for this test.
E, nu, mu1, mu2, c_coupling = par_nucleus_init_viscoelastic(msh)
I = ufl.Identity(gdim)

_dt = fem.Constant(msh, PETSc.ScalarType(1e-2))


def eps(w):
    return ufl.sym(ufl.grad(w))

def phi(w):
    return ufl.tr(eps(w))

V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
Q = fem.functionspace(msh, ("Lagrange", 1))     


u_inner = fem.Function(V, name="u_inner")
u_outer = fem.Function(V, name="u_outer")

u_inner.x.array[:] = 0.0
u_inner.x.scatter_forward()

u_inner, u_inner_prev, p, p_prev = init_state_nucleus(V, Q)

v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

dx = dx_n          
ds = ds_n          




sigma_e = 2 * (E / (2 * (1 + nu))) * eps(u_inner) + (E * nu / ((1 + nu) * (1 - 2 * nu))) * ufl.tr(eps(u_inner)) * I
sigma_v = (mu1 / _dt) * (eps(u_inner) - eps(u_inner_prev)) + (mu2 / _dt) * (phi(u_inner) - phi(u_inner_prev)) * I
sigma = sigma_e + sigma_v

# --- weak form: div(sigma)=0 in Ω_n, ---
F_vol = ufl.inner(sigma, ufl.grad(v)) * dx
F_rhs = c_coupling * p * ufl.div(v) * dx
# --- Neumann BC: traction = 0 
zero_flux = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0)))
bcs = []
F_bc = 0.0

bcs_n, F_bc_n = bc_u.set_bc_inner_neumann(zero_flux, v, ds)
F_bc += F_bc_n

# regularizer 
eps_reg = fem.Constant(msh, PETSc.ScalarType(1e-6))
dx_tagged = ufl.Measure("dx", domain=msh, subdomain_data=ct)
F_reg = eps_reg * ufl.inner(u_inner, v) * dx_tagged  

F = F_vol + F_bc + F_reg - F_rhs
J = ufl.derivative(F, u_inner, du)

problem = NonlinearProblem(F, u_inner, bcs=bcs, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

T_final = 1.0
num_steps = 50
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"[NUCLEUS] dt = {dt_value}, num_steps = {num_steps}")

def tag(t: float) -> str:
    return f"{t:.6f}"

with io.XDMFFile(MPI.COMM_WORLD, "data/nucleus_viscoelastic.xdmf", "w") as xdmf:
    msh.name = "mesh_at_t0.000000"
    xdmf.write_mesh(msh)

    u_inner.name = "u"
    xdmf.write_function(u_inner, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value

        u_inner_prev.x.array[:] = u_inner.x.array
        u_inner_prev.x.scatter_forward()

        n_it, converged = solver.solve(u_inner)

        u_inner.x.scatter_forward()

        msh.geometry.x[:] += u_inner.x.array.reshape((-1, gdim))

        msh.name = f"mesh_at_t{tag(t)}"
        xdmf.write_mesh(msh)
        xdmf.write_function(u_inner, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

        u_inner_prev.x.array[:] = u_inner.x.array[:]
        u_inner.x.array[:] = 0.0
        u_inner_prev.x.scatter_forward()
        u_inner.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:
    print("Written nucleus_viscoelastic.xdmf")