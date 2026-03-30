from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem, io, mesh as dmesh
import ufl

from src.mesh_initialization import build_spherical_shell_mesh,build_ball_mesh
from src.parameter_init import init_state, par_cytoplasm_init

# -----------------------------
# Build mesh + tags
# -----------------------------
#msh, ct, ft, dx, ds_outer, ds_inner = build_spherical_shell_mesh()
msh, ct, ft, dx, ds_outer = build_ball_mesh(R=1.0, lc=0.3)
tdim = msh.topology.dim
gdim = msh.geometry.dim


outer_id = 11
def move_mesh_like_dealii_simple(msh, u, u_prev, gdim):
    # works for vector P1 on a P1 mesh
    du = u.x.array.reshape((-1, gdim)) - u_prev.x.array.reshape((-1, gdim))

    x = msh.geometry.x
    cell_vertices = msh.geometry.dofmap
    touched = np.zeros(x.shape[0], dtype=bool)

    for verts in cell_vertices:
        # simple surrogate of deal.II quadrature-sum motion
        v_disp = du[verts].sum(axis=0)

        for vid in verts:
            if not touched[vid]:
                x[vid, :gdim] += v_disp
                touched[vid] = True

################################################################################
# Initializing constants
######################################
E, nu, mu1, mu2, _ = par_cytoplasm_init(msh) 
I = ufl.Identity(gdim)
_dt = fem.Constant(msh, PETSc.ScalarType(0.001))

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
u, u_prev, *_ = init_state(V, Q)

v  = ufl.TestFunction(V)

A = fem.Function(Q, name="A")
M = fem.Function(Q, name="M")
A_prev = fem.Function(Q, name="A_prev")
M_prev = fem.Function(Q, name="M_prev")
u_out = fem.Function(V, name="u")

qA = ufl.TestFunction(Q)
dA = ufl.TrialFunction(Q)
qM = ufl.TestFunction(Q)
dM = ufl.TrialFunction(Q)

# Chemistry params (same defaults as before; tune later)
da  = fem.Constant(msh, PETSc.ScalarType(0.01))
dm  = fem.Constant(msh, PETSc.ScalarType(0.001))
ka  = fem.Constant(msh, PETSc.ScalarType(0.04))
kam = fem.Constant(msh, PETSc.ScalarType(0.06))
kma = fem.Constant(msh, PETSc.ScalarType(0.05))
ac  = fem.Constant(msh, PETSc.ScalarType(1.0))
mc  = fem.Constant(msh, PETSc.ScalarType(1.0))
Kc  = fem.Constant(msh, PETSc.ScalarType(1.0))

pr   = fem.Constant(msh, PETSc.ScalarType(1.7))
psi  = fem.Constant(msh, PETSc.ScalarType(200))
c_my = fem.Constant(msh, PETSc.ScalarType(-40.0))
asat = fem.Constant(msh, PETSc.ScalarType(1.4))


K_react = fem.Constant(msh, PETSc.ScalarType(1.0))

def S_term(Af, Mf):
    num = (Af * Af) * (mc - Mf)
    den = 1.0 + (K_react / kma) * (Af * Af)
    return kam * num / den
R_A = ka * (ac - A_prev) + S_term(A_prev, M_prev)
R_M = -kma * (ac - A_prev) - S_term(A_prev, M_prev)

# -----------------------------
# Reaction–diffusion (semi-implicit)
# (A^{n+1},q) + dt*da*(grad A^{n+1},grad q) = (A^n,q) + dt*(R_A(A^n,M^n),q)
# same for M
# -----------------------------
a_lhs_form = ufl.inner(dA, qA) * dx + _dt * da * ufl.inner(ufl.grad(dA), ufl.grad(qA)) * dx
a_rhs_form = ufl.inner(A_prev, qA) * dx + _dt * ufl.inner(R_A, qA) * dx

m_lhs_form = ufl.inner(dM, qM) * dx + _dt * dm * ufl.inner(ufl.grad(dM), ufl.grad(qM)) * dx
m_rhs_form = ufl.inner(M_prev, qM) * dx + _dt * ufl.inner(R_M, qM) * dx

a_lhs = fem.form(a_lhs_form)
a_rhs = fem.form(a_rhs_form)
m_lhs = fem.form(m_lhs_form)
m_rhs = fem.form(m_rhs_form)

A_mat = fem.petsc.assemble_matrix(a_lhs)
A_mat.assemble()
M_mat = fem.petsc.assemble_matrix(m_lhs)
M_mat.assemble()

bA = fem.petsc.create_vector(a_rhs)
bM = fem.petsc.create_vector(m_rhs)

kspA = PETSc.KSP().create(MPI.COMM_WORLD)
kspA.setType("cg")
kspA.getPC().setType("sor")
kspA.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
kspA.setOperators(A_mat)

kspM = PETSc.KSP().create(MPI.COMM_WORLD)
kspM.setType("cg")
kspM.getPC().setType("sor")
kspM.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
kspM.setOperators(M_mat)

# -----------------------------
# Deal.II-style pressure parameters
# p = pr*(1 + 2/pi*atan(A)*delta)/(1+dilation) + c*M + psi*exp(-A/asat)*A^2
# -----------------------------


DG0 = fem.functionspace(msh, ("DG", 0))
delta = fem.Function(DG0, name="delta")
delta.x.array[:] = 0.0

# Ensure facet->cell connectivity exists
msh.topology.create_connectivity(tdim - 1, tdim)
f2c = msh.topology.connectivity(tdim - 1, tdim)

# Find facets on outer boundary: ft == 11
outer_facets = ft.indices[ft.values == outer_id]

# Mark adjacent cells
for f in outer_facets:
    cells = f2c.links(f)
    for c in cells:
        delta.x.array[c] = 1.0

delta.x.scatter_forward()


dilation = ufl.div(u_prev)
den = dilation+1.0#ufl.max_value(1.0 + dilation, 1e-3)

p1 = (
    pr * (1.0 + (2.0 / ufl.pi) * ufl.atan(A_prev) * delta) / den
    + c_my * M_prev
    + psi * ufl.exp(-A_prev / asat) * (A_prev * A_prev)
)
n = ufl.FacetNormal(msh)

eps_reg = fem.Constant(msh, PETSc.ScalarType(1e-16))

# ---- linear mechanics: a(u,v) = L(v)
u_trial = ufl.TrialFunction(V)

sigma_e_trial = (E / (1 + nu)) * eps(u_trial) + (E * nu / ((1 + nu) * (1 - 2 * nu))) * ufl.tr(eps(u_trial)) * I
sigma_v_trial = (mu1 / _dt) * eps(u_trial) + (mu2 / _dt) * phi(u_trial) * I
sigma_trial = sigma_e_trial + sigma_v_trial

sigma_v_known = (mu1 / _dt) * eps(u_prev) + (mu2 / _dt) * phi(u_prev) * I

a_form = ufl.inner(sigma_trial, ufl.grad(v)) * dx + eps_reg * ufl.inner(u_trial, v) * dx

L_form = (
    ufl.inner(sigma_v_known, ufl.grad(v)) * dx
    - p1 * ufl.div(v) * dx
    #- p1 * ufl.dot(v, n) * (ds_outer + ds_inner)
    + p1 * ufl.dot(v, n) * ds_outer 
)

a = fem.form(a_form)
L = fem.form(L_form)

K = fem.petsc.assemble_matrix(a)
K.assemble()
b = fem.petsc.create_vector(L)

ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setType("gmres")
ksp.getPC().setType("none")
ksp.setTolerances(rtol=1.3e-4, atol=1e-14, max_it=5000)
ksp.setOperators(K)
# -----------------------------
# Time stepping
# -----------------------------
T_final = 67
num_steps = 1000
dt_value = 0.005
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"dt = {dt_value}, num_steps = {num_steps}")

# init mechanics
u.x.array[:] = 0.0
u_prev.x.array[:] = 0.0
u.x.scatter_forward()
u_prev.x.scatter_forward()



# init chemistry: deal.II-like initial conditions
def hash01(x, y, z, seed=0.0):
    """
    Deterministic pseudo-random number in [0,1) from coordinates.
    Better than np.random here because it is reproducible and MPI-safe.
    """
    s = np.sin(12.9898 * x + 78.233 * y + 37.719 * z + seed) * 43758.5453123
    return s - np.floor(s)

def dealii_mode(coords, k=2.08158):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    r = np.sqrt(x*x + y*y + z*z)
    mode = np.zeros_like(r)

    mask = r > 0.05
    rm = r[mask]
    kr = k * rm

    radial_part = ((3.0 / (k * k * rm * rm) - 1.0) * np.sin(kr) / (kr)
                   - 3.0 * np.cos(kr) / (k * k * rm * rm))
    angular_part = (-x[mask] * x[mask] - y[mask] * y[mask] + 2.0 * z[mask] * z[mask]) / (rm * rm)

    mode[mask] = radial_part * angular_part
    mode = np.nan_to_num(mode, nan=0.0, posinf=0.0, neginf=0.0)
    mode = np.clip(mode, -10.0, 10.0)
    return mode

def set_dealii_like_initial_conditions(Q, A, M, A_prev, M_prev):
    coords = Q.tabulate_dof_coordinates().reshape((-1, gdim))
    mode = dealii_mode(coords, k=2.08158)

    # independent deterministic "random" factors for A and M
    randA = hash01(coords[:, 0], coords[:, 1], coords[:, 2], seed=0.123)
    randM = hash01(coords[:, 0], coords[:, 1], coords[:, 2], seed=4.567)

    A0 = 1.0 + 0.1 * mode * randA
    M0 = 1.0 - 0.1 * mode * randM

    A.x.array[:] = A0
    M.x.array[:] = M0
    A_prev.x.array[:] = A0
    M_prev.x.array[:] = M0

    A.x.scatter_forward()
    M.x.scatter_forward()
    A_prev.x.scatter_forward()
    M_prev.x.scatter_forward()


set_dealii_like_initial_conditions(Q, A, M, A_prev, M_prev)

def tag(t: float) -> str:
    return f"{t:.6f}"

out_path = "data/shell_mech_plus_AM_with_pressure.xdmf"

with io.XDMFFile(MPI.COMM_WORLD, out_path, "w") as xdmf:
    msh.name = "mesh_at_t0.000000"
    xdmf.write_mesh(msh)

    u_out.x.array[:] = u.x.array
    u_out.x.scatter_forward()

    #xdmf.write_function(u_out, 0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    xdmf.write_function(A,     0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    #xdmf.write_function(M,     0.0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

    for k in range(num_steps):
        t = (k + 1) * dt_value
        A_prev.x.array[:] = A.x.array
        M_prev.x.array[:] = M.x.array
        A_prev.x.scatter_forward()
        M_prev.x.scatter_forward()
        # ---- mechanics (linear KSP)
        u_prev.x.array[:] = u.x.array
        u_prev.x.scatter_forward()
        A_old = A.x.array.copy()
        M_old = M.x.array.copy()
        K.zeroEntries()
        fem.petsc.assemble_matrix(K, a)
        K.assemble()
        ksp.setOperators(K)

        with b.localForm() as loc:
            loc.set(0.0)
        fem.petsc.assemble_vector(b, L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
        ksp.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"[mech-linear] step {k+1}/{num_steps}, t={t:.6f}, KSP iters={ksp.getIterationNumber()}")


        # ---- chemistry RHS on OLD mesh
        with bA.localForm() as loc:
            loc.set(0.0)
        fem.petsc.assemble_vector(bA, a_rhs)
        bA.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
               mode=PETSc.ScatterMode.REVERSE)

        with bM.localForm() as loc:
            loc.set(0.0)
        fem.petsc.assemble_vector(bM, m_rhs)
        bM.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
               mode=PETSc.ScatterMode.REVERSE)

        # ---- ALE mesh update
        # move_mesh_like_dealii_simple(msh, u, u_prev, gdim)
        # msh.geometry.x[:] += (
        #     u.x.array.reshape((-1, gdim)) - u_prev.x.array.reshape((-1, gdim))
        # )

        # ---- chemistry LHS on NEW mesh
        A_mat.zeroEntries()
        fem.petsc.assemble_matrix(A_mat, a_lhs)
        A_mat.assemble()
        kspA.setOperators(A_mat)

        M_mat.zeroEntries()
        fem.petsc.assemble_matrix(M_mat, m_lhs)
        M_mat.assemble()
        kspM.setOperators(M_mat)

        # ---- solve chemistry
        kspA.solve(bA, A.x.petsc_vec)
        kspM.solve(bM, M.x.petsc_vec)
        A.x.scatter_forward()
        M.x.scatter_forward()


        if MPI.COMM_WORLD.rank == 0:
            print("dA max =", np.max(np.abs(A.x.array - A_old)))
            print("dM max =", np.max(np.abs(M.x.array - M_old)))

        if MPI.COMM_WORLD.rank == 0:
            print(f"[chem old-RHS/new-LHS] A iters={kspA.getIterationNumber()}, M iters={kspM.getIterationNumber()}")

        # ---- output
        u_out.x.array[:] = u.x.array
        u_out.x.scatter_forward()

        A_prev.x.array[:] = A.x.array
        M_prev.x.array[:] = M.x.array
        A_prev.x.scatter_forward()
        M_prev.x.scatter_forward()
        if MPI.COMM_WORLD.rank == 0:
            umax = np.max(np.linalg.norm(u.x.array.reshape((-1, gdim)), axis=1))
            print(f"max|u| = {umax:.6e}")
            print(f"A range = [{A.x.array.min():.6e}, {A.x.array.max():.6e}]")
            print(f"M range = [{M.x.array.min():.6e}, {M.x.array.max():.6e}]")
        msh.name = f"mesh_at_t{tag(t)}"
        
        xdmf.write_mesh(msh)
        #xdmf.write_function(u_out, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
        xdmf.write_function(A,     t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
        #xdmf.write_function(M,     t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
        print("||b_mech|| =", b.norm())
        print("mech converged reason =", ksp.getConvergedReason())  
if MPI.COMM_WORLD.rank == 0:
    print(f"Written {out_path} (u + A + M with deal.II-style pressure)")
if MPI.COMM_WORLD.rank == 0:
    print("unique cell tags:", np.unique(ct.values))
    print("unique facet tags:", np.unique(ft.values))
    print("num nodes =", msh.geometry.x.shape[0])
    print("num cells =", msh.topology.index_map(msh.topology.dim).size_local)