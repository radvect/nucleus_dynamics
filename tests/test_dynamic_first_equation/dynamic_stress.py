from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem, io
from dolfinx.mesh import create_submesh
import ufl

from src.mesh_initialization import build_two_region_ball_mesh

# --- геометрия и меры ---
msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()

ids = np.unique(ct.values)
print("Cell tags:", ids)

outer_id = 1  # цитоплазма
inner_id = 2  # ядро
print(f"outer_id (cyto) = {outer_id}, inner_id (nucleus) = {inner_id}")

tdim = msh.topology.dim
gdim = msh.geometry.dim

cells_inner = ct.indices[ct.values == inner_id]
cells_outer = ct.indices[ct.values == outer_id]

res_inner = create_submesh(msh, tdim, cells_inner)
res_outer = create_submesh(msh, tdim, cells_outer)

msh_inner = res_inner[0]
msh_outer = res_outer[0]

# --- пространства ---
V_inner = fem.functionspace(msh_inner, ("Lagrange", 1, (gdim,)))
V_outer = fem.functionspace(msh_outer, ("Lagrange", 1, (gdim,)))

# ядро: u = 0 (просто для вывода)
u_inner = fem.Function(V_inner, name="u_inner")

def u_inner_expr(x):
    return np.vstack((0.0 * x[0],
                      0.0 * x[1],
                      0.0 * x[2]))

u_inner.interpolate(u_inner_expr)

# цитоплазма: динамика u(t)
u = fem.Function(V_outer, name="u_outer")      # u^{n+1}
u_prev = fem.Function(V_outer, name="u_prev")  # u^{n}

v = ufl.TestFunction(V_outer)
du = ufl.TrialFunction(V_outer)
dx = ufl.dx(domain=msh_outer)

# --- желаемое поле на внешней сфере (Dirichlet) ---
u_bc_fun = fem.Function(V_outer, name="u_bc")

def u_exact_expr(x):
    # "вращение" вокруг z: u = (-y, x, 0)
    return np.vstack((
        -x[1],
        x[0],
        0.0 * x[2],
    ))

u_bc_fun.interpolate(u_exact_expr)

# --- граничное условие Дирихле на мембране |x| = R_outer ---
R_outer = 1.0  # должен совпадать с build_two_region_ball_mesh

def outer_boundary(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return np.isclose(r, R_outer, atol=1e-8)

dofs_outer = fem.locate_dofs_geometrical(V_outer, outer_boundary)
bc_outer = fem.dirichletbc(u_bc_fun, dofs_outer)
bcs = [bc_outer]

# --- материалы: упругость ---
def eps(w):
    return ufl.sym(ufl.grad(w))

I = ufl.Identity(gdim)

mu_val = 1.0   # параметр Ламе μ (упругость)
lam_val = 1.0  # параметр Ламе λ
mu = fem.Constant(msh_outer, PETSc.ScalarType(mu_val))
lam = fem.Constant(msh_outer, PETSc.ScalarType(lam_val))

# --- вязкие параметры (динамика) ---
mu1_val = 1.0  # "сдвиговая" вязкость
mu2_val = 1.0  # объёмная вязкость

mu1 = fem.Constant(msh_outer, PETSc.ScalarType(mu1_val))
mu2 = fem.Constant(msh_outer, PETSc.ScalarType(mu2_val))

# --- шаг по времени и число шагов ---
T_final = 1.0
num_steps = 10
dt_value = T_final / num_steps
dt = fem.Constant(msh_outer, PETSc.ScalarType(dt_value))

print(f"dt = {dt_value}, num_steps = {num_steps}")

# --- P и связь c_coupling ---
Q_outer = fem.functionspace(msh_outer, ("Lagrange", 1))
P = fem.Function(Q_outer, name="P")

def P_expr(x):
    # пример: линейное поле по z, P = z
    return x[2]

P.interpolate(P_expr)

c_coupling = fem.Constant(msh_outer, PETSc.ScalarType(1.0))

# --- тензоры деформаций и объемные деформации ---
eps_u = eps(u)
eps_prev = eps(u_prev)

phi_u = ufl.tr(eps_u)       # φ(u^{n+1})
phi_prev = ufl.tr(eps_prev) # φ(u^{n})

# упругая часть σ_e(u^{n+1})
sigma_e = 2 * mu * eps_u + lam * ufl.tr(eps_u) * I

# вязкая часть σ_v (backward Euler по времени)
sigma_v = (
    mu1 / dt * (eps_u - eps_prev)
    + mu2 / dt * (phi_u - phi_prev) * I
)

# полный тензор напряжений
sigma = sigma_e + sigma_v

# --- слабая форма ---
# ∫ sigma : grad(v) dx
F_vol = ufl.inner(sigma, ufl.grad(v)) * dx

# источник от P: c_coupling * P * ∂_z v_z
F_rhs = c_coupling * P * ufl.grad(v)[2, 2] * dx

F = F_vol - F_rhs

# маленький объёмный регулятор для rigid-body modes
eps_reg = fem.Constant(msh_outer, PETSc.ScalarType(1e-8))
F += eps_reg * ufl.inner(u, v) * dx

J = ufl.derivative(F, u, du)

# --- задача и солвер (одни на весь временной цикл) ---
problem = NonlinearProblem(F, u, bcs=bcs, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

# начальное условие: пусть внутри всё ноль
u.x.array[:] = 0.0
u_prev.x.array[:] = 0.0

# --- временной цикл ---
with io.XDMFFile(MPI.COMM_WORLD, "nucleus_test.xdmf", "w") as xdmf_n:
    xdmf_n.write_mesh(msh_inner)
    xdmf_n.write_function(u_inner)

# --- запись цитоплазмы как time series ---
with io.XDMFFile(MPI.COMM_WORLD, "cytoplasm_time_series.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_outer)
    # начальное состояние t=0
    xdmf.write_function(u, t=0.0)

    for n in range(num_steps):
        t = (n + 1) * dt_value
        if MPI.COMM_WORLD.rank == 0:
            print(f"\nTime step {n+1}/{num_steps}, t = {t:.3f}")

        # u_prev = u^n
        u_prev.x.array[:] = u.x.array

        def u_bc_expr_time(x):
            a = np.sin(2.0 * np.pi * t / T_final)
            return np.vstack((
                -a * x[1],
                a * x[0],
                0.0 * x[2],
            ))

        u_bc_fun.interpolate(u_bc_expr_time)

        # решить на шаге t^{n+1}
        n_it, converged = solver.solve(u)
        u.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"  Newton iterations: {n_it}, converged = {converged}")

        # записать состояние на время t
        xdmf.write_function(u, t=t)

if MPI.COMM_WORLD.rank == 0:
    print("Written nucleus_test.xdmf and cytoplasm_time_series.xdmf")