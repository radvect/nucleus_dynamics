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
from src.parameter_init import init_state, par_cytoplasm_init


#building meshes
msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()

#parameter initialization
E, nu, mu1, mu2, c_coupling = par_cytoplasm_init(msh)
nu_prime = nu / (1.0 - 2.0 * nu) #for convinience usage
I = ufl.Identity(msh.topology.dim)
_dt = fem.Constant(msh, PETSc.ScalarType(1e-2))
def eps(w):
    return ufl.sym(ufl.grad(w))

def phi(w):
    return ufl.tr(eps(w))


ids = np.unique(ct.values)
print("Cell tags:", ids)

outer_id = 1  
inner_id = 2  
print(f"outer_id (cyto) = {outer_id}, inner_id (nucleus) = {inner_id}")

tdim = msh.topology.dim 
gdim = msh.geometry.dim

cells_inner = ct.indices[ct.values == inner_id]
cells_outer = ct.indices[ct.values == outer_id]

res_inner = create_submesh(msh, tdim, cells_inner)
res_outer = create_submesh(msh, tdim, cells_outer)

msh_inner = res_inner[0]
msh_outer = res_outer[0]

# function initializing 
V_inner = fem.functionspace(msh_inner, ("Lagrange", 1, (gdim,)))
V_outer = fem.functionspace(msh_outer, ("Lagrange", 1, (gdim,)))
Q_inner = fem.functionspace(msh_inner, ("Lagrange", 1))
Q_outer = fem.functionspace(msh_outer, ("Lagrange", 1))


u_inner = fem.Function(V_inner, name="u_inner")
u_outer = fem.Function(V_outer, name="u_outer")

u_inner.x.array[:] = 0.0
u_inner.x.scatter_forward()


# temporary comment - we want to initialize nucleus in the similar way as the cytoplasm
#u_inner, u_inner_prev, p, p_prev = init_state(V_inner, Q_inner)

u_outer, u_outer_prev, P, P_prev = init_state(V_outer, Q_outer)



u_bc_fun = fem.Function(V_outer, name="u_bc")

def u_exact_expr(x):
    return np.vstack((
        -x[1],
        x[0],
        0.0 * x[2],
    ))

u_bc_fun.interpolate(u_exact_expr)







v = ufl.TestFunction(V_outer)
du = ufl.TrialFunction(V_outer)

##CYTOPLASM SOLVING: u = u_outer
u = u_outer
u_prev = u_outer_prev
dx = dx_c

sigma_e = 2 * E/2/(1+nu) * eps(u) + E*nu/(1+nu)/(1-2*nu) * ufl.tr(eps(u)) * I
sigma_v = mu1/_dt * (eps(u) - eps(u_prev)) + mu2/_dt * (phi(u) - phi(u_prev)) * I
sigma = sigma_e + sigma_v

#weak form

F_vol = ufl.inner(sigma, ufl.grad(v)) * dx
F_rhs = c_coupling * P * ufl.grad(v)[2, 2] * dx
F = F_vol - F_rhs
J = ufl.derivative(F, u, du)

problem = NonlinearProblem(F, u, bcs=bcs, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

n_it, converged = solver.solve(u)
u.x.scatter_forward()



# R_outer = 1.0 

# def outer_boundary(x):
#     r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#     return np.isclose(r, R_outer, atol=1e-8)

# dofs_outer = fem.locate_dofs_geometrical(V_outer, outer_boundary)


# bc_outer = fem.dirichletbc(u_bc_fun, dofs_outer)


# bcs = [bc_outer]




if MPI.COMM_WORLD.rank == 0:
    print(f"Newton iterations: {n_it}, converged = {converged}")


with io.XDMFFile(MPI.COMM_WORLD, "nucleus_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_inner)
    xdmf.write_function(u_inner)

with io.XDMFFile(MPI.COMM_WORLD, "cytoplasm_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh_outer)
    xdmf.write_function(u)

if MPI.COMM_WORLD.rank == 0:
    print("Written nucleus_test.xdmf and cytoplasm_test.xdmf")
