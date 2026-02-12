from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from src.mesh_initialization import build_two_region_ball_mesh
from src.parameter_init import par_nucleus_init_viscoelastic, init_state_nucleus
import src.boundary_setup_displacement as bc_u


# -----------------------------
# Mesh + tags
# -----------------------------
msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()
tdim = msh.topology.dim
gdim = msh.geometry.dim

outer_id = 1
inner_id = 2

# -----------------------------
# Parameters (nucleus viscoelastic)
# -----------------------------
E, nu, mu1, mu2, c_coupling = par_nucleus_init_viscoelastic(msh)
I = ufl.Identity(gdim)
_dt = fem.Constant(msh, PETSc.ScalarType(1e-2))

def eps(w):
    return ufl.sym(ufl.grad(w))

def phi(w):
    return ufl.tr(eps(w))

# -----------------------------
# Function spaces + nucleus state
# -----------------------------
V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
Q = fem.functionspace(msh, ("Lagrange", 1))

u_inner, u_inner_prev, p, p_prev = init_state_nucleus(V, Q)

v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

dx = dx_n   # nucleus volume
ds = ds_n   # nucleus interface

# -----------------------------
# Build a DOF mask for CYTOPLASM vertices (CG1 vector space)
# We'll freeze cytoplasm: u = 0 on dofs belonging to outer region vertices
# Mark vertex as cytoplasm if it touches at least one cytoplasm cell (ct==outer_id)
# -----------------------------
msh.topology.create_connectivity(0, tdim)
v2c = msh.topology.connectivity(0, tdim)

imap = V.dofmap.index_map
n_local = imap.size_local
n_ghost = imap.num_ghosts
n_total = n_local + n_ghost

vertex_in_cytoplasm = np.zeros(n_total, dtype=bool)
for vtx in range(n_local):
    cells = v2c.links(vtx)
    if cells.size and np.any(ct.values[cells] == outer_id):
        vertex_in_cytoplasm[vtx] = True

# safer on partition boundaries: keep ghost vertices "on" to avoid holes
if n_ghost:
    vertex_in_cytoplasm[n_local:] = True

bs = V.dofmap.bs
assert bs == gdim
dof_in_cytoplasm = np.repeat(vertex_in_cytoplasm, bs)

# -----------------------------
# Constitutive law in nucleus
# -----------------------------
sigma_e = (E / (1 + nu)) * eps(u_inner) + (E * nu / ((1 + nu) * (1 - 2 * nu))) * ufl.tr(eps(u_inner)) * I
sigma_v = (mu1 / _dt) * (eps(u_inner) - eps(u_inner_prev)) + (mu2 / _dt) * (phi(u_inner) - phi(u_inner_prev)) * I
sigma = sigma_e + sigma_v

# Weak form in Ω_n (nucleus only)
F_vol = ufl.inner(sigma, ufl.grad(v)) * dx
F_rhs = c_coupling * p * ufl.div(v) * dx

# Neumann traction = 0 on interface
zero_flux = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0)))
bcs = []
F_bc = 0.0
_, F_bc_n = bc_u.set_bc_inner_neumann(zero_flux, v, ds)
F_bc += F_bc_n

# Regularizer (tiny), over whole domain (as in your code)
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

# -----------------------------
# Time stepping
# -----------------------------
T_final = 5.0
num_steps = 50
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"[NUCLEUS] dt = {dt_value}, num_steps = {num_steps}")

def tag(t: float) -> str:
    return f"{t:.6f}"

out_path = "data/nucleus_viscoelastic_fixed_cytoplasm.xdmf"

with io.XDMFFile(MPI.COMM_WORLD, out_path, "w") as xdmf:
    msh.name = "mesh_at_t0.000000"
    xdmf.write_mesh(msh)

    u_inner.name = "u"
    # freeze cytoplasm at t=0 too
    u_inner.x.array[dof_in_cytoplasm] = 0.0
    u_inner.x.scatter_forward()
    xdmf.write_function(u_inner, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value

        # store previous
        u_inner_prev.x.array[:] = u_inner.x.array
        u_inner_prev.x.scatter_forward()

        # solve
        n_it, converged = solver.solve(u_inner)
        u_inner.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"[NUCLEUS] step {k+1}/{num_steps}, t={t:.6f}, Newton iters={n_it}, converged={converged}")

        # *** hard-freeze CYTOPLASM dofs (outer region) ***
        u_inner.x.array[dof_in_cytoplasm] = 0.0
        u_inner.x.scatter_forward()

        # update geometry incrementally (nucleus moves, cytoplasm vertices don't)
        msh.geometry.x[:] += (
            u_inner.x.array.reshape((-1, gdim)) - u_inner_prev.x.array.reshape((-1, gdim))
        )

        msh.name = f"mesh_at_t{tag(t)}"
        xdmf.write_mesh(msh)
        xdmf.write_function(u_inner, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

        # reset (your pattern)
        u_inner_prev.x.array[:] = u_inner.x.array[:]
        u_inner.x.array[:] = 0.0
        # keep cytoplasm fixed in the reset state too (safe)
        u_inner.x.array[dof_in_cytoplasm] = 0.0

        u_inner_prev.x.scatter_forward()
        u_inner.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:
    print(f"Written {out_path}")
