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
import src.boundary_setup_displacement as bc_u
import src.boundary_setup_pressure as bc_p

#building meshes
msh, ct, ft, dx_c, dx_n, ds_c, ds_n = build_two_region_ball_mesh()
tdim = msh.topology.dim 
gdim = msh.geometry.dim
#parameter initialization
E, nu, mu1, mu2, c_coupling = par_cytoplasm_init(msh)
nu_prime = nu / (1.0 - 2.0 * nu) #for convinience usage
I = ufl.Identity(gdim)
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

# function initializing 
V = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))  
Q = fem.functionspace(msh, ("Lagrange", 1))     


u_inner = fem.Function(V, name="u_inner")
u_outer = fem.Function(V, name="u_outer")

u_inner.x.array[:] = 0.0
u_inner.x.scatter_forward()


# temporary comment - we want to initialize nucleus in the similar way as the cytoplasm
#u_inner, u_inner_prev, p, p_prev = init_state(V_inner, Q_inner)

u_outer, u_outer_prev, P, P_prev = init_state(V, Q)


v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

##BC
R_outer = 1.0
R_inner = 0.4

def is_boundary_close(x, R):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return np.isclose(r, R, atol=1e-8)


dofs_outer = fem.locate_dofs_geometrical(V, lambda x: is_boundary_close(x, R_outer))
dofs_inner = fem.locate_dofs_geometrical(V, lambda x: is_boundary_close(x, R_inner))


bcs = []
F_bc=0

n = ufl.FacetNormal(msh)


##CYTOPLASM SOLVING: u = u_outer
u = u_outer
u_prev = u_outer_prev
dx = dx_c

sigma_e = 2 * E/2/(1+nu) * eps(u) + E*nu/(1+nu)/(1-2*nu) * ufl.tr(eps(u)) * I
sigma_v = mu1/_dt * (eps(u) - eps(u_prev)) + mu2/_dt * (phi(u) - phi(u_prev)) * I
sigma = sigma_e + sigma_v

#weak form
#Currently, we preset P as a known function. Therefore, we don't extract any BC for the Pressure P part. If needed, boundary_setup_pressure.py exists

F_vol = ufl.inner(sigma, ufl.grad(v)) * dx
F_rhs = c_coupling * P * ufl.div(v) * dx


zero_flux = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0)))
bcs_inner, F_bc_inner = bc_u.set_bc_inner_neumann(zero_flux, v, ds_n)
bcs_outer, F_bc_outer = bc_u.set_bc_outer_neumann(zero_flux, v, ds_c)

'''TODO: finish bcs logic'''
#bcs +=bcs_inner 
#bcs +=bcs_outer


F_bc+=F_bc_outer
F_bc+=F_bc_inner

'''TODO: Pressure currently not in use. P equations should be added later'''
'''IMPORTANT - F_rhs should be changed as well'''
# F_bc += bc_p.set_bc_inner_neumann(P,n, v, ds_n)
# F_bc += bc_p.set_bc_outer_neumann(P,n, v, ds_c)

#regularizer
eps_reg = fem.Constant(msh, PETSc.ScalarType(1e-6)) 
dx_tagged = ufl.Measure("dx", domain=msh, subdomain_data=ct)
F_reg = eps_reg * ufl.inner(u, v) * dx_tagged

F = F_vol - F_rhs + F_bc + F_reg
J = ufl.derivative(F, u, du)

problem = NonlinearProblem(F, u, bcs=bcs, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
ksp = solver.krylov_solver
print("KSP prefix:", ksp.getOptionsPrefix())

solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

# --- time stepping ---
solver.rtol = 1e-8
solver.atol = 1e-10
solver.max_it = 25

# --- time stepping ---
T_final = 1.0
num_steps = 10
dt_value = T_final / num_steps
_dt.value = PETSc.ScalarType(dt_value)

if MPI.COMM_WORLD.rank == 0:
    print(f"dt = {dt_value}, num_steps = {num_steps}")

# initial condition
u.x.array[:] = 0.0
u_prev.x.array[:] = 0.0
u.x.scatter_forward()
u_prev.x.scatter_forward()

with io.XDMFFile(MPI.COMM_WORLD, "data/"+"nucleus_test.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_inner)

with io.XDMFFile(MPI.COMM_WORLD, "data/"+"cytoplasm_time_series.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u, t=0.0)

    for k in range(num_steps):
        t = (k + 1) * dt_value
        if MPI.COMM_WORLD.rank == 0:
            print(f"Time step {k+1}/{num_steps}, t = {t}")

        u_prev.x.array[:] = u.x.array
        u_prev.x.scatter_forward()

        n_it, converged = solver.solve(u)
        u.x.scatter_forward()

        if MPI.COMM_WORLD.rank == 0:
            print(f"Newton iterations: {n_it}, converged = {converged}")

        xdmf.write_function(u, t=t)

if MPI.COMM_WORLD.rank == 0:
    print("Written nucleus_test.xdmf and cytoplasm_time_series.xdmf")