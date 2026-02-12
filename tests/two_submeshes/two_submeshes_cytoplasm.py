from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl

from dolfinx import fem, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.mesh import create_submesh

from src.mesh_initialization import build_two_region_ball_mesh
from src.parameter_init import par_cytoplasm_init, init_state, init_state_cyto_test


msh, ct, ft, dx_c_parent, dx_n_parent, ds_c_parent, ds_n_parent = build_two_region_ball_mesh()
tdim = msh.topology.dim
gdim = msh.geometry.dim

outer_id = 1  # cytoplasm
inner_id = 2  # nucleus

cyto_cells = ct.indices[ct.values == outer_id]
nucleus_cells = ct.indices[ct.values == inner_id]

if MPI.COMM_WORLD.rank == 0:
    print(f"[PARENT] #cells cyto={len(cyto_cells)}, nucleus={len(nucleus_cells)}")

sub_c = create_submesh(msh, tdim, cyto_cells)
sub_n = create_submesh(msh, tdim, nucleus_cells)
msh_c = sub_c[0]
msh_n = sub_n[0]

# ---------- spaces ----------
Vc = fem.functionspace(msh_c, ("Lagrange", 1, (gdim,)))
Qc = fem.functionspace(msh_c, ("Lagrange", 1))

Vn = fem.functionspace(msh_n, ("Lagrange", 1, (gdim,)))  # viz only
Qn = fem.functionspace(msh_n, ("Lagrange", 1))           # viz only

# ---------- init states ----------
u_c, u_c_prev, P_c, P_c_prev = init_state_cyto_test(Vc, Qc, R_inner=0.4, R_outer=1.0, amp=1.0)

u_n = fem.Function(Vn, name="u_nucleus")
u_n.x.array[:] = 0.0
u_n.x.scatter_forward()

p_n = fem.Function(Qn, name="p_nucleus")
p_n.x.array[:] = 0.0
p_n.x.scatter_forward()

# ---------- material params on cytoplasm submesh ----------
E, nu, mu1, mu2, c_coupling = par_cytoplasm_init(msh_c)
I = ufl.Identity(gdim)
_dt = fem.Constant(msh_c, PETSc.ScalarType(1e-2))

def eps(w): return ufl.sym(ufl.grad(w))
def phi(w): return ufl.tr(eps(w))

dx_c = ufl.Measure("dx", domain=msh_c)

v = ufl.TestFunction(Vc)
du = ufl.TrialFunction(Vc)

mu = E / (2 * (1 + nu))
lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

sigma_e = 2 * mu * eps(u_c) + lmbda * ufl.tr(eps(u_c)) * I
sigma_v = (mu1 / _dt) * (eps(u_c) - eps(u_c_prev)) + (mu2 / _dt) * (phi(u_c) - phi(u_c_prev)) * I
sigma = sigma_e + sigma_v

F_vol = ufl.inner(sigma, ufl.grad(v)) * dx_c
F_rhs = c_coupling * P_c * ufl.div(v) * dx_c

eps_reg = fem.Constant(msh_c, PETSc.ScalarType(1e-6))
F_reg = eps_reg * ufl.inner(u_c, v) * dx_c

F = F_vol - F_rhs + F_reg
J = ufl.derivative(F, u_c, du)

problem = NonlinearProblem(F, u_c, bcs=[], J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

# ---------- time stepping ----------
T_final = 1.0
num_steps = 1
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"dt = {dt_value}, num_steps = {num_steps}")

def tag(t: float) -> str:
    return f"{t:.6f}"

# ---------- write XDMF (save BOTH u and p) ----------
with io.XDMFFile(MPI.COMM_WORLD, "data/cytoplasm_submesh_time_series.xdmf", "w") as xdmf_c, \
     io.XDMFFile(MPI.COMM_WORLD, "data/nucleus_submesh_zero.xdmf", "w") as xdmf_n:

    msh_c.name = "cyto_mesh_t0"
    msh_n.name = "nucleus_mesh_t0"
    xdmf_c.write_mesh(msh_c)
    xdmf_n.write_mesh(msh_n)

    u_c.name = "u_cyto"
    P_c.name = "p_cyto"
    #xdmf_c.write_function(u_c, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")
    xdmf_c.write_function(P_c, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")

    u_n.name = "u_nucleus"
    p_n.name = "p_nucleus"
    xdmf_n.write_function(u_n, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")
    xdmf_n.write_function(p_n, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value

        # time history
        u_c_prev.x.array[:] = u_c.x.array
        u_c_prev.x.scatter_forward()
        P_c_prev.x.array[:] = P_c.x.array
        P_c_prev.x.scatter_forward()

        n_it, converged = solver.solve(u_c)
        u_c.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"[step {k+1}/{num_steps}] it={n_it}, converged={converged}")

        # If P_c is prescribed (not solved), keep it as-is; otherwise update it elsewhere.

        msh_c.name = f"cyto_mesh_t{tag(t)}"
        xdmf_c.write_mesh(msh_c)
        #xdmf_c.write_function(u_c, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")
        xdmf_c.write_function(P_c, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")

        msh_n.name = f"nucleus_mesh_t{tag(t)}"
        xdmf_n.write_mesh(msh_n)
        xdmf_n.write_function(u_n, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")
        xdmf_n.write_function(p_n, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")

if MPI.COMM_WORLD.rank == 0:
    print("Saved:")
    print("  data/cytoplasm_submesh_time_series.xdmf  (u_cyto + p_cyto)")
    print("  data/nucleus_submesh_zero.xdmf          (u_nucleus + p_nucleus)")
