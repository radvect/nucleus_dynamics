import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, LinearProblem
import meshio
import gmsh

# --- GMSH MESH ---
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

model = gmsh.model()
model.add("circle")

circle = model.occ.addCircle(0, 0, 0, 1)
loop = model.occ.addCurveLoop([circle])
surface = model.occ.addPlaneSurface([loop])

model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

model.add_physical_group(dim=1, tags=[circle], name="Boundary")
model.add_physical_group(dim=2, tags=[surface], name="Interior")

model.mesh.generate(dim=2)
gmsh.write("circle.msh")
gmsh.finalize()

mesh1 = meshio.read("circle.msh")
triangle_mesh = meshio.Mesh(points=mesh1.points, cells={"triangle": mesh1.cells_dict["triangle"]})
meshio.write("circle.xdmf", triangle_mesh)

# --- LOAD MESH INTO DOLFINX ---
with io.XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

V = fem.functionspace(domain, ("Lagrange", 1))

# --- DIRICHLET BOUNDARY CONDITION p = 0 ---
boundary_facets = mesh.locate_entities_boundary(domain, dim=1, marker=lambda x: np.full(x.shape[1], True))
zero = fem.Constant(domain, PETSc.ScalarType(0.0))
bc = fem.dirichletbc(zero, fem.locate_dofs_topological(V, 1, boundary_facets), V)

# --- INITIAL CONDITIONS ---
def initial_pressure(x, a=-1, b=1):
    return np.exp(-a * (x[0]**2 + b * x[1]**2))

def laplacian_initial_pressure(x, a=-1, b=1):
    return 4 * a * b * (a * x[0]**2 + b * x[1]**2 - 1) * np.exp(-a * (x[0]**2 + b * x[1]**2))

p_n = fem.Function(V)
p_n.name = "p"
p_n.interpolate(initial_pressure)

phi_n = fem.Function(V)
phi_n.name = "phi"
phi_n.interpolate(laplacian_initial_pressure)

# --- SOURCE TERM (f) ---
f = fem.Function(V)
f.interpolate(lambda x: np.zeros(x.shape[1]))
# НЕ ЗАДАЁМ ИМЯ И НЕ ПИШЕМ f В ФАЙЛ

phi, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, v1 = ufl.TrialFunction(V), ufl.TestFunction(V)

dt = 1.0 / 100
k = 1
C = 1

a = C * phi * v * ufl.dx + k * dt * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
L = (phi_n - dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)

solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A)
solver_phi.setType(PETSc.KSP.Type.PREONLY)
solver_phi.getPC().setType(PETSc.PC.Type.LU)

# --- XDMF OUTPUT ---
xdmf = io.XDMFFile(domain.comm, "pressure_map_2.xdmf", "w")
xdmf.write_mesh(domain)

p_h = fem.Function(V)
p_h.name = "p_h"
p_h.interpolate(initial_pressure)
xdmf.write_function(p_h, 0.0)

p_n.x.array[:] = p_h.x.array

xdmf_velocity = io.XDMFFile(domain.comm, "velocity_map_2.xdmf", "w")
xdmf_velocity.write_mesh(domain)

U = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim,)))

def compute_velocity(p_func):
    u_ufl = -ufl.as_vector([ufl.grad(p_func)[0], ufl.grad(p_func)[1]])
    return fem.Expression(u_ufl, U.element.interpolation_points())

u_function = fem.Function(U)
u_function.name = "u"
u_function.interpolate(compute_velocity(p_n))
xdmf_velocity.write_function(u_function, 0.0)

t = 0.0
num_steps = 40

# --- INITIAL PRESSURE SOLVE ---
a2 = -ufl.dot(ufl.grad(p), ufl.grad(v1)) * ufl.dx
L2 = phi_n * v1 * ufl.dx
problem = LinearProblem(a2, L2, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
p_n = problem.solve()
p_n.name = "p"

# Нормировка в [0, 1]
local_min = np.min(p_n.x.array)
local_max = np.max(p_n.x.array)
global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
global_max = domain.comm.allreduce(local_max, op=MPI.MAX)
range_p = global_max - global_min + 1e-14
p_n.x.array[:] = (p_n.x.array - global_min) / range_p

print(f"p {p_n.x.array[:]}, {t}")
xdmf.write_function(p_n, t)

# --- TIME LOOP ---
for step in range(num_steps):
    t += dt
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    solver_phi.solve(b, phi_n.x.petsc_vec)
    phi_n.x.scatter_forward()
    print(f"phi {phi_n.x.array[:]}, {t}")

    a2 = -ufl.dot(ufl.grad(p), ufl.grad(v1)) * ufl.dx
    L2 = phi_n * v1 * ufl.dx
    problem = LinearProblem(a2, L2, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p_n = problem.solve()
    p_n.name = "p"

    # Нормировка давления в [0, 1]
    local_min = np.min(p_n.x.array)
    local_max = np.max(p_n.x.array)
    global_min = domain.comm.allreduce(local_min, op=MPI.MIN)
    global_max = domain.comm.allreduce(local_max, op=MPI.MAX)
    range_p = global_max - global_min + 1e-14
    p_n.x.array[:] = (p_n.x.array - global_min) / range_p

    print(f"p {p_n.x.array[:]}, {t}")

    u_function.interpolate(compute_velocity(p_n))
    xdmf.write_function(p_n, t)
    xdmf_velocity.write_function(u_function, t)

xdmf.close()
xdmf_velocity.close()
