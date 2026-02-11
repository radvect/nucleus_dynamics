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
from src.parameter_init import par_nucleus_init_viscoelastic, init_state_nucleus, init_state
import src.boundary_setup_displacement as bc_u


# ----------------------------
# Build parent (two-region) mesh
# ----------------------------
msh, ct, ft, dx_c_parent, dx_n_parent, ds_c_parent, ds_n_parent = build_two_region_ball_mesh()
tdim = msh.topology.dim
gdim = msh.geometry.dim

# region ids (make sure these match your build_two_region_ball_mesh tagging)
outer_id = 1  # cytoplasm
inner_id = 2  # nucleus

# Extract cell indices for each region
cell_indices = ct.indices
cell_values = ct.values

nucleus_cells = cell_indices[cell_values == inner_id]
cyto_cells = cell_indices[cell_values == outer_id]

if MPI.COMM_WORLD.rank == 0:
    print(f"[PARENT] #cells nucleus={len(nucleus_cells)}, cyto={len(cyto_cells)}")

# ----------------------------
# Create two submeshes
# ----------------------------
# create_submesh returns: (submesh, entity_map, vertex_map, geom_map) depending on dolfinx version
sub_n = create_submesh(msh, tdim, nucleus_cells)
sub_c = create_submesh(msh, tdim, cyto_cells)

msh_n = sub_n[0]
msh_c = sub_c[0]

# Optional maps (handy later if you want to transfer fields)
# entity_map_n = sub_n[1]
# entity_map_c = sub_c[1]

# ----------------------------
# Nucleus material parameters + time step
# ----------------------------
E, nu, mu1, mu2, c_coupling = par_nucleus_init_viscoelastic(msh_n)
I = ufl.Identity(gdim)

_dt = fem.Constant(msh_n, PETSc.ScalarType(1e-2))

def eps(w): return ufl.sym(ufl.grad(w))
def phi(w): return ufl.tr(eps(w))

# ----------------------------
# Function spaces on each submesh
# ----------------------------
Vn = fem.functionspace(msh_n, ("Lagrange", 1, (gdim,)))
Qn = fem.functionspace(msh_n, ("Lagrange", 1))

Vc = fem.functionspace(msh_c, ("Lagrange", 1, (gdim,)))  # for viz only
Qc = fem.functionspace(msh_c, ("Lagrange", 1))           # for viz only

# ----------------------------
# State init (NUCLEUS ONLY drives dynamics here)
# ----------------------------
u_n, u_n_prev, p_n, p_n_prev = init_state_nucleus(Vn, Qn)

# Cytoplasm fields: keep ~0 just for visualization (you can later solve on msh_c if needed)
u_c = fem.Function(Vc, name="u_cyto")
p_c = fem.Function(Qc, name="p_cyto")
u_c.x.array[:] = 0.0
p_c.x.array[:] = 0.0
u_c.x.scatter_forward()
p_c.x.scatter_forward()

v = ufl.TestFunction(Vn)
du = ufl.TrialFunction(Vn)

# Measures on nucleus submesh
dx_n = ufl.Measure("dx", domain=msh_n)
ds_n = ufl.Measure("ds", domain=msh_n)  # boundary of nucleus submesh (this is the interface boundary in parent)

# ----------------------------
# Constitutive law (Kelvin–Voigt style)
# ----------------------------
mu = E / (2 * (1 + nu))
lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

sigma_e = 2 * mu * eps(u_n) + lmbda * ufl.tr(eps(u_n)) * I
sigma_v = (mu1 / _dt) * (eps(u_n) - eps(u_n_prev)) + (mu2 / _dt) * (phi(u_n) - phi(u_n_prev)) * I
sigma = sigma_e + sigma_v

# ----------------------------
# Weak form on nucleus submesh
#   - equilibrium: div(sigma)=0
#   - pressure coupling: - c_coupling * p * div(v)
#   - Neumann traction: set to 0 (free boundary) OR replace by pressure traction if you want
# ----------------------------
F_vol = ufl.inner(sigma, ufl.grad(v)) * dx_n
F_rhs = c_coupling * p_n * ufl.div(v) * dx_n  # your original volume coupling

# traction = 0 on nucleus boundary (submesh boundary)
zero_flux = fem.Constant(msh_n, PETSc.ScalarType((0.0, 0.0, 0.0)))
bcs = []
F_bc = 0.0
_, F_bc_n = bc_u.set_bc_inner_neumann(zero_flux, v, ds_n)
F_bc += F_bc_n

# Regularizer ONLY on nucleus (avoid touching cytoplasm DOF, since we are on submesh anyway)
eps_reg = fem.Constant(msh_n, PETSc.ScalarType(1e-6))
F_reg = eps_reg * ufl.inner(u_n, v) * dx_n

F = F_vol + F_bc + F_reg - F_rhs
J = ufl.derivative(F, u_n, du)

problem = NonlinearProblem(F, u_n, bcs=bcs, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

# ----------------------------
# Time stepping
# ----------------------------
T_final = 0.01
num_steps = 100
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"[NUCLEUS-submesh] dt = {dt_value}, num_steps = {num_steps}")

def l2_norm_global(vec: np.ndarray) -> float:
    loc2 = np.dot(vec, vec)
    glob2 = MPI.COMM_WORLD.allreduce(loc2, op=MPI.SUM)
    return float(np.sqrt(glob2))

def tag(t: float) -> str:
    return f"{t:.6f}"

# Write two separate XDMF files (cleanest for ParaView)
with io.XDMFFile(MPI.COMM_WORLD, "data/nucleus_submesh_viscoelastic.xdmf", "w") as xdmf_n, \
     io.XDMFFile(MPI.COMM_WORLD, "data/cytoplasm_submesh_zero.xdmf", "w") as xdmf_c:

    # t=0
    msh_n.name = "nucleus_mesh_t0"
    msh_c.name = "cyto_mesh_t0"
    xdmf_n.write_mesh(msh_n)
    xdmf_c.write_mesh(msh_c)

    u_n.name = "u_nucleus"
    p_n.name = "p_nucleus"
    xdmf_n.write_function(u_n, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")
    xdmf_n.write_function(p_n, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")

    u_c.name = "u_cyto"
    xdmf_c.write_function(u_c, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value

        # Store previous nucleus displacement (for viscosity)
        u_n_prev.x.array[:] = u_n.x.array
        u_n_prev.x.scatter_forward()

        n_it, converged = solver.solve(u_n)
        u_n.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"[step {k+1}/{num_steps}] Newton it={n_it}, converged={converged}, ||u_n||_2={l2_norm_global(u_n.x.array):.6e}")

        # Optional diagnostics: pressure in nucleus
        p_int = fem.assemble_scalar(fem.form(p_n * dx_n))
        p_int = MPI.COMM_WORLD.allreduce(p_int, op=MPI.SUM)
        if MPI.COMM_WORLD.rank == 0:
            print(f"  ∫_Ωn p dx = {p_int:.6e}")

        # Move ONLY nucleus submesh geometry (cytoplasm submesh stays unchanged)
        msh_n.geometry.x[:] += u_n.x.array.reshape((-1, gdim))

        # Write at time t
        msh_n.name = f"nucleus_mesh_t{tag(t)}"
        xdmf_n.write_mesh(msh_n)
        xdmf_n.write_function(u_n, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")
        xdmf_n.write_function(p_n, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_n.name}']")

        # cytoplasm stays ~0 (write same zero field just for timeline consistency if you want)
        msh_c.name = f"cyto_mesh_t{tag(t)}"
        xdmf_c.write_mesh(msh_c)
        xdmf_c.write_function(u_c, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh_c.name}']")

if MPI.COMM_WORLD.rank == 0:
    print("Written:")
    print("  data/nucleus_submesh_viscoelastic.xdmf")
    print("  data/cytoplasm_submesh_zero.xdmf")
