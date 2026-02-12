from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem, io
import ufl

from src.mesh_initialization import build_two_region_ball_mesh
from src.parameter_init import init_state, par_cytoplasm_init
import src.boundary_setup_displacement as bc_u


# -----------------------------
# Build mesh + tags
# -----------------------------
msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()
tdim = msh.topology.dim
gdim = msh.geometry.dim

outer_id = 1
inner_id = 2

# -----------------------------
# Parameters
# -----------------------------
E, nu, mu1, mu2, c_coupling = par_cytoplasm_init(msh)
I = ufl.Identity(gdim)
_dt = fem.Constant(msh, PETSc.ScalarType(1e-2))

def eps(w):
    return ufl.sym(ufl.grad(w))

def phi(w):
    return ufl.tr(eps(w))

if MPI.COMM_WORLD.rank == 0:
    print("Cell tags:", np.unique(ct.values))

# -----------------------------
# Function spaces + state
# -----------------------------
V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
Q = fem.functionspace(msh, ("Lagrange", 1))
u, u_prev, P, P_prev = init_state(V, Q)

v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

# -----------------------------
# Build a DOF mask for nucleus vertices (CG1 vector space)
# Mark vertex as nucleus if it touches at least one nucleus cell (ct==inner_id)
# Then expand to vector dofs
# -----------------------------
msh.topology.create_connectivity(0, tdim)
v2c = msh.topology.connectivity(0, tdim)

imap = V.dofmap.index_map
n_local = imap.size_local
n_ghost = imap.num_ghosts
n_total = n_local + n_ghost

vertex_in_nucleus = np.zeros(n_total, dtype=bool)
for vtx in range(n_local):
    cells = v2c.links(vtx)
    if cells.size and np.any(ct.values[cells] == inner_id):
        vertex_in_nucleus[vtx] = True

# safer for partition boundaries: keep ghost vertices "on" (avoid holes)
if n_ghost:
    vertex_in_nucleus[n_local:] = True

bs = V.dofmap.bs
assert bs == gdim
dof_in_nucleus = np.repeat(vertex_in_nucleus, bs)

# We'll write u_out where nucleus dofs are zero
u_out = fem.Function(V, name="u")  # name "u" for ParaView convenience

# -----------------------------
# Variational problem (cytoplasm only)
# -----------------------------
dx = dx_c  # integrate ONLY over cytoplasm cells

sigma_e = (E / (1 + nu)) * eps(u) + (E * nu / ((1 + nu) * (1 - 2 * nu))) * ufl.tr(eps(u)) * I
sigma_v = (mu1 / _dt) * (eps(u) - eps(u_prev)) + (mu2 / _dt) * (phi(u) - phi(u_prev)) * I
sigma = sigma_e + sigma_v

F_vol = ufl.inner(sigma, ufl.grad(v)) * dx
F_rhs = c_coupling * P * ufl.div(v) * dx

zero_flux = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0)))
_, F_bc_inner = bc_u.set_bc_inner_neumann(zero_flux, v, ds_n)
_, F_bc_outer = bc_u.set_bc_outer_neumann(zero_flux, v, ds_c)
F_bc = F_bc_inner + F_bc_outer

eps_reg = fem.Constant(msh, PETSc.ScalarType(1e-6))
dx_tagged = ufl.Measure("dx", domain=msh, subdomain_data=ct)
F_reg = eps_reg * ufl.inner(u, v) * dx_tagged

F = F_vol - F_rhs + F_bc + F_reg
J = ufl.derivative(F, u, du)

problem = NonlinearProblem(F, u, bcs=[], J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

# -----------------------------
# Time stepping
# -----------------------------
T_final = 1.0
num_steps = 10
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"dt = {dt_value}, num_steps = {num_steps}")

u.x.array[:] = 0.0
u_prev.x.array[:] = 0.0
u.x.scatter_forward()
u_prev.x.scatter_forward()

def tag(t: float) -> str:
    return f"{t:.6f}"

out_path = "data/cytoplasm_only_u.xdmf"

with io.XDMFFile(MPI.COMM_WORLD, out_path, "w") as xdmf:
    msh.name = "mesh_at_t0.000000"
    xdmf.write_mesh(msh)

    # write t=0 (u_out = u with nucleus zero)
    u_out.x.array[:] = u.x.array
    u_out.x.array[dof_in_nucleus] = 0.0
    u_out.x.scatter_forward()
    xdmf.write_function(u_out, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value

        u_prev.x.array[:] = u.x.array
        u_prev.x.scatter_forward()

        n_it, converged = solver.solve(u)
        u.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"step {k+1}/{num_steps}, t={t:.6f}, Newton iters={n_it}, converged={converged}")

        # update mesh geometry incrementally
        msh.geometry.x[:] += (
            u.x.array.reshape((-1, gdim)) - u_prev.x.array.reshape((-1, gdim))
        )

        # prepare output field: zero nucleus DOFs
        u_out.x.array[:] = u.x.array
        u_out.x.array[dof_in_nucleus] = 0.0
        u_out.x.scatter_forward()

        msh.name = f"mesh_at_t{tag(t)}"
        xdmf.write_mesh(msh)
        xdmf.write_function(u_out, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

        # reset (your pattern)
        u_prev.x.array[:] = u.x.array[:]
        u.x.array[:] = 0.0
        u_prev.x.scatter_forward()
        u.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:
    print(f"Written {out_path} (single function, nucleus DOFs zeroed each step)")
