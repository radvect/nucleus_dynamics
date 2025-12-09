from __future__ import annotations
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from src.mesh_initialization import build_two_region_ball_mesh
import ufl
from ufl import inner, grad, sym, Identity
from dolfinx import fem, mesh, io
from dolfinx.fem import petsc
from src.parameter_init import par_init
from dolfinx.nls.petsc import NewtonSolver

# ----------------------------------------------------------------------
# 1. Сетка и параметры
# ----------------------------------------------------------------------
# Build mesh and measures (двухобластной шар: цитоплазма + ядро)
msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()

# init params
dS = ufl.Measure("dS", domain=msh, subdomain_data=ft)
dS_n = dS(12)  # пока не используем в F, но оставим как есть

I, E, nu, mu1, mu2, c_coupling, zeta, eta_f, lam_s, mu_s, _dt, nu_prime = par_init(msh)

# ----------------------------------------------------------------------
# 2. Пространства
# ----------------------------------------------------------------------
Vs = fem.functionspace(msh, ("CG", 1))          # scalar H1
Vv = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))  # vector H1

# Cytoplasm variables
V_u = Vv
V_P = Vs

u = fem.Function(V_u, name="u")      # смещение цитоплазмы
P = fem.Function(V_P, name="P")      # поле давления/активности

# пока P = const = 1
P_const = fem.Constant(msh, PETSc.ScalarType(1.0))
P.interpolate(fem.Expression(P_const, V_P.element.interpolation_points()))

v_u = ufl.TestFunction(V_u)
u_n = fem.Function(V_u, name="u_n")
u_n.x.array[:] = 0.0

# Nucleus variables (Omega_n) пока НЕ используем
# V_phi_s = Vs
# V_vs = Vv
# V_vf = Vv
# V_p = Vs
# V_us = Vv
#
# phi_s = fem.Function(V_phi_s, name="phi_s")
# vs = fem.Function(V_vs, name="v_s")
# vf = fem.Function(V_vf, name="v_f")
# p = fem.Function(V_p, name="p")
# us = fem.Function(V_us, name="u_s")
#
# q_phi = ufl.TestFunction(V_phi_s)
# w_s = ufl.TestFunction(V_vs)
# w_f = ufl.TestFunction(V_vf)
# q_p = ufl.TestFunction(V_p)
# w_us = ufl.TestFunction(V_us)

# ----------------------------------------------------------------------
# 3. Конститутивный закон (Kelvin–Voigt)
# ----------------------------------------------------------------------
C11 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
C12 = E * nu          / ((1.0 + nu) * (1.0 - 2.0 * nu))
C33 = E / (2.0 * (1.0 + nu))

D11 = mu1 + mu2
D12 = mu2
D33 = mu1 / 2.0
cCoupl = 1

# Ламе-коэффициенты из C**, D**
mu_e  = C33      # = E / (2*(1+nu))
lam_e = C12      # = E*nu / ((1+nu)*(1-2*nu))
mu_v  = D33      # = mu1/2
lam_v = D12      # = mu2

def eps(w):
    return sym(grad(w))

def epsdot(w):
    return sym(grad((w - u_n) / _dt))

# Полное напряжение: упругое + вязкое
sigma_e = 2 * mu_e * eps(u)    + lam_e * ufl.tr(eps(u))    * I
sigma_v = 2 * mu_v * epsdot(u) + lam_v * ufl.tr(epsdot(u)) * I
sigma   = sigma_e + sigma_v

v = v_u  # тестовая функция (вектор)

# ----------------------------------------------------------------------
# 4. Слабая форма (цитоплазма)
# ----------------------------------------------------------------------
# Объёмный вклад ∫_Ωc σ : ∇v dx
F_vol = inner(sigma, grad(v)) * dx_c

# Источник от P (как у тебя: действует через grad(v)[2,2] в объёме)
F_rhs = c_coupling * P * grad(v)[2, 2] * dx_c

# Базовый резидуал
F = F_vol - F_rhs

# Маленький объемный регулятор ε ∫ u·v dx (прибивает rigid-body modes слегка)
eps_reg = fem.Constant(msh, PETSc.ScalarType(1e-8))
F += eps_reg * inner(u, v) * dx_c

du = ufl.TrialFunction(V_u)
J = ufl.derivative(F, u, du)

# --- время для вязкой части ---
_dt.value = PETSc.ScalarType(1e-2)
u_n.x.array[:] = 0.0

# ----------------------------------------------------------------------
# 5. Граничные условия Дирихле = 0
#    - снаружи (внешняя сфера)
#    - внутри (весь объём ядра)
# ----------------------------------------------------------------------
tdim = msh.topology.dim
fdim = tdim - 1

# --- внешняя граница: u = 0 на внешней сфере ---
x = msh.geometry.x
R_outer = np.max(np.linalg.norm(x, axis=1))

def outer_boundary(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return np.isclose(r, R_outer, rtol=1e-3, atol=1e-3)

facets_outer = mesh.locate_entities_boundary(msh, fdim, outer_boundary)
dofs_outer = fem.locate_dofs_topological(V_u, fdim, facets_outer)

u_bc_outer = fem.Function(V_u)
u_bc_outer.x.array[:] = 0.0
bc_outer = fem.dirichletbc(u_bc_outer, dofs_outer)

# --- объём ядра: u = 0 в Ω_n ---
tdim = msh.topology.dim

# обязательно строим коннективность 3->3, иначе locate_dofs_topological ругается
msh.topology.create_connectivity(tdim, tdim)

ids = np.unique(ct.values)
print("cell tags:", ids)  # можешь один раз посмотреть в выводе, кто есть кто
nucleus_id = ids[0]       # или поставь руками правильный id ядра

cells_nucleus = ct.indices[ct.values == nucleus_id]
dofs_nucleus = fem.locate_dofs_topological(V_u, tdim, cells_nucleus)

u_bc_nucleus = fem.Function(V_u)
u_bc_nucleus.x.array[:] = 0.0
bc_nucleus = fem.dirichletbc(u_bc_nucleus, dofs_nucleus)

bcs = [bc_outer, bc_nucleus]

# ----------------------------------------------------------------------
# 6. Постановка задачи и решение
# ----------------------------------------------------------------------
problem = petsc.NonlinearProblem(fem.form(F), u, bcs=bcs, J=fem.form(J))
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.atol = 1e-10
solver.rtol = 1e-10
solver.max_it = 25

solver.solve(u)

with io.XDMFFile(MPI.COMM_WORLD, "u_stress_dirichlet.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    
    # --- сохранить метки ячеек ---
    msh.topology.create_connectivity(msh.topology.dim, 0)  # иногда нужно для корректности
    xdmf.write_meshtags(ct, msh.geometry)  # <<< ВАЖНО
    
    # --- сохранить решение ---
    xdmf.write_function(u)