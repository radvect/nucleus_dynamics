from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import fem, io, nls
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

# ядро: u = 0
u_inner = fem.Function(V_inner, name="u_inner")

def u_inner_expr(x):
    return np.vstack((0.0 * x[0],
                      0.0 * x[1],
                      0.0 * x[2]))

u_inner.interpolate(u_inner_expr)

# цитоплазма: решаем вариационную задачу
u = fem.Function(V_outer, name="u_outer")
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

# --- материалы / σ(u) ---
def eps(w):
    return ufl.sym(ufl.grad(w))

I = ufl.Identity(gdim)

mu_val = 1.0
lam_val = 1.0
mu = fem.Constant(msh_outer, PETSc.ScalarType(mu_val))
lam = fem.Constant(msh_outer, PETSc.ScalarType(lam_val))

sigma = 2 * mu * eps(u) + lam * ufl.tr(eps(u)) * I

# --- P и связь c_coupling ---
# P можешь потом заменить на своё поле
Q_outer = fem.functionspace(msh_outer, ("Lagrange", 1))
P = fem.Function(Q_outer, name="P")

def P_expr(x):
    # пример: линейное поле по z
    return x[2]

P.interpolate(P_expr)

c_coupling = fem.Constant(msh_outer, PETSc.ScalarType(1.0))

# --- слабая форма ---
# ∫ sigma : grad(v) dx_c
F_vol = ufl.inner(sigma, ufl.grad(v)) * dx

# источник от P: c_coupling * P * ∂_z v_z
F_rhs = c_coupling * P * ufl.grad(v)[2, 2] * dx

F = F_vol - F_rhs

# маленький объёмный регулятор для rigid-body modes

J = ufl.derivative(F, u, du)




R_outer = 1.0  # радиус внешнего шара (должен совпадать с тем, что в build_two_region_ball_mesh)

def outer_boundary(x):
    # x имеет форму (3, N): x[0] = x, x[1] = y, x[2] = z
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    # возвращаем булев массив: True там, где точка на сфере радиуса R_outer
    return np.isclose(r, R_outer, atol=1e-8)

# дофы (степени свободы в V_outer), для которых геометрические координаты удовлетворяют outer_boundary
dofs_outer = fem.locate_dofs_geometrical(V_outer, outer_boundary)

# --- 3) Собираем граничное условие Дирихле ---
bc_outer = fem.dirichletbc(u_bc_fun, dofs_outer)

# Список всех ГУ (сейчас только одно)
bcs = [bc_outer]




# --- решаем (без граничных условий, просто как есть) ---
problem = NonlinearProblem(F, u, bcs=bcs, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

n_it, converged = solver.solve(u)
u.x.scatter_forward()

if MPI.COMM_WORLD.rank == 0:
    print(f"Newton iterations: {n_it}, converged = {converged}")

# --- вывод ---
with io.XDMFFile(MPI.COMM_WORLD, "nucleus_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_inner)
    xdmf.write_function(u_inner)

with io.XDMFFile(MPI.COMM_WORLD, "cytoplasm_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_outer)
    xdmf.write_function(u)

if MPI.COMM_WORLD.rank == 0:
    print("Written nucleus_test.xdmf and cytoplasm_test.xdmf")
