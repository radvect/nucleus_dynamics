import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector
import meshio

t = 0.0  
T = 1.0 
num_steps = 50 
dt = T / num_steps  

mesh1 = meshio.read("ellypse.msh")

triangle_mesh = meshio.Mesh(
    points=mesh1.points,
    cells={"triangle": mesh1.cells_dict["triangle"]}
)
meshio.write("ellypse_preprocessed.xdmf", triangle_mesh)

with io.XDMFFile(MPI.COMM_WORLD, "ellypse_preprocessed.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

V = fem.functionspace(domain, ("Lagrange", 1))

def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

u_n = fem.Function(V)
u_n.interpolate(initial_condition)

def source_function(x):
    return 0*np.exp(np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))

f = fem.Function(V)
f.interpolate(source_function)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

k = 1
C = 1

a = C*u * v * ufl.dx + k*dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n - dt * f) * v * ufl.dx

#Neumann destroys the third term
bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

xdmf = io.XDMFFile(domain.comm, "nucleus_neumann.xdmf", "w")
xdmf.write_mesh(domain)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

for step in range(num_steps):
    t += dt

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    u_n.x.array[:] = uh.x.array

    xdmf.write_function(uh, t)

xdmf.close()