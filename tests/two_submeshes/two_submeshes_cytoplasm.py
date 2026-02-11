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
from src.parameter_init import par_cytoplasm_init, init_state


msh, ct, ft, dx_c_parent, dx_n_parent, ds_c_parent, ds_n_parent = build_two_region_ball_mesh()
tdim = msh.topology.dim
gdim = msh.geometry.dim

outer_id = 1  # cytoplasm
inner_id = 2  # nucleus

cell_indices = ct.indices
cell_values  = ct.values

cyto_cells    = cell_indices[cell_values == outer_id]
nucleus_cells = cell_indices[cell_values == inner_id]

if MPI.COMM_WORLD.rank == 0:
    print(f"[PARENT] #cells cyto={len(cyto_cells)}, nucleus={len(nucleus_cells)}")


sub_c = create_submesh(msh, tdim, cyto_cells)
sub_n = create_submesh(msh, tdim, nucleus_cells)

msh_c = sub_c[0]
msh_n = sub_n[0]


E, nu, mu1, mu2, c_coupling = par_cytoplasm_init(msh_c)
I = ufl.Identity(gdim)

_dt = fem.Constant(msh_c, PETSc.ScalarType(1e-2))

def eps(w): return ufl.sym(ufl.grad(w))
def phi(w): return ufl.tr(eps(w))


Vc = fem.functionspace(msh_c, ("Lagrange", 1, (gdim,)))
Qc = fem.functionspace(msh_c, ("Lagrange", 1))

Vn = fem.functionspace(msh_n, ("Lagrange", 1, (gdim,)))  # for viz only
Qn = fem.functionspace(msh_n, ("Lagrange", 1))           # for viz only

#Cytoplasm state 
u_c, u_c_prev, P_c, P_c_prev = init_state(Vc, Qc)

# Nucleus (zero)
u_n = fem.Function(Vn, name="u_nucleus")
u_n.x.array[:] = 0.0
u_n.x.scatter_forward()


dx_c = ufl.Measure("dx", domain=msh_c)
# ds_c = ufl.Measure("ds", domain=msh_c)  

v = ufl.TestFunction(Vc)
du = ufl.TrialFunction(Vc)

mu = E / (2 * (1 + nu))
lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

sigma_e = 2 * mu * eps(u_c) + lmbda * ufl.tr(eps(u_c)) * I
sigma_v = (mu1 / _dt) * (eps(u_c) - eps(u_c_prev)) + (mu2 / _dt) * (phi(u_c) - phi(u_c_prev)) * I
sigma = sigma_e + sigma_v

F_vol = ufl.inner(sigma, ufl.grad(v)) * dx_c
F_rhs = c_coupling * P_c * ufl.div(v) * dx_c

F_bc = 0.0


eps_reg = fem.Constant(msh_c, PETSc.ScalarType(1e-6))
F_reg = eps_reg * ufl.inner(u_c, v) * dx_c

F = F_vol - F_rhs + F_bc + F_reg
J = ufl.derivative(F, u_c, du)

problem = NonlinearProblem(F, u_c, bcs=[], J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25


# Time stepping
T_final = 1.0
num_steps = 50
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"dt = {dt_value}, num_steps = {num_steps}")

def tag(t: float) -> str:
    return f"{t:.6f}"



with io.XDMFFile(MPI.COMM_WORLD, "data/cytoplasm_submesh_time_series.xdmf", "w") as xdmf_c, \
     io.XDMFFile(MPI.COMM_WORLD, "data/nucleus_submesh_zero.xdmf", "w") as xdmf_n:

    msh_c.name = "cyto_mesh_t0"
    msh_n.name = "nucleus_mesh_t0"
    xdmf_c.write_mesh(msh_c)
    xdmf_n.write_mesh(msh_n)

    u_c.name = "u_cyto"
    xdmf_c.write_function(u_c, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")

    u_n.name = "u_nucleus"
    xdmf_n.write_function(u_n, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value

        u_c_prev.x.array[:] = u_c.x.array
        u_c_prev.x.scatter_forward()

        n_it, converged = solver.solve(u_c)
        u_c.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"[step {k+1}/{num_steps}] it={n_it}")


        msh_c.geometry.x[:] += u_c.x.array.reshape((-1, gdim))

        msh_c.name = f"cyto_mesh_t{tag(t)}"
        xdmf_c.write_mesh(msh_c)
        xdmf_c.write_function(u_c, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")

        msh_n.name = f"nucleus_mesh_t{tag(t)}"
        xdmf_n.write_mesh(msh_n)
        xdmf_n.write_function(u_n, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")

        u_c_prev.x.array[:] = u_c.x.array[:]
        u_c.x.array[:] = 0.0
        u_c_prev.x.scatter_forward()
        u_c.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:
    print("  data/cytoplasm_submesh_time_series.xdmf")
    print("  data/nucleus_submesh_zero.xdmf")
