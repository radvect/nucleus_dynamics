import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import meshio
from dolfinx.fem.petsc import LinearProblem
try:
    import gmsh  
except ImportError:
    print("This script requires gmsh to be installed")
    exit(0)


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
triangle_mesh = meshio.Mesh(
    points=mesh1.points,
    cells={"triangle": mesh1.cells_dict["triangle"]}
)
meshio.write("circle.xdmf", triangle_mesh)


with io.XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

V = fem.functionspace(domain, ("Lagrange", 1))




def initial_pressure(x, a=-1, b=1):
    return np.exp(-a * (x[0]**2 + b * x[1]**2))
def laplacian_initial_pressure(x, a=-1, b=1):
    return 4 * a * b * (a * x[0]**2 + b * x[1]**2 - 1) * np.exp(-a * (x[0]**2 + b * x[1]**2))

p_n = fem.Function(V)
p_n.interpolate(initial_pressure)


phi_n = fem.Function(V)
phi_n.interpolate(laplacian_initial_pressure)
print(phi_n.x.array[:])


def source_function(x):
    return 0*np.exp(np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))

f = fem.Function(V)
f.interpolate(source_function)

phi, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, v1 = ufl.TrialFunction(V), ufl.TestFunction(V)

dt = 1.0 / 100
k = 1
C = 1
C1 = 1

a = C*phi * v * ufl.dx + k*dt * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
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

a2 = ufl.dot(ufl.grad(p), ufl.grad(v1)) * ufl.dx
L2 = phi_n * v1 * ufl.dx

# bilinear_form_Poisson = fem.form(a2)
# linear_form_Poisson = fem.form(L2)

# A_P = assemble_matrix(bilinear_form_Poisson)
# A_P.assemble()
# b_P = create_vector(linear_form_Poisson)

# solver_p = PETSc.KSP().create(domain.comm)
# solver_p.setOperators(A_P)
# solver_p.setType(PETSc.KSP.Type.PREONLY)
# solver_p.getPC().setType(PETSc.PC.Type.LU)

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
num_steps = 10

dx = ufl.dx(domain)
phi_avg = fem.assemble_scalar(fem.form(phi_n * dx)) / domain.comm.allreduce(domain.geometry.volume, op=MPI.SUM)
phi_n.x.array[:] -= phi_avg


a2 = -ufl.dot(ufl.grad(p), ufl.grad(v1)) * ufl.dx
L2 = phi_n * v1 * ufl.dx
problem = LinearProblem(a2, L2, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
p_n = problem.solve()
print(f"p {p_n.x.array[:]}, {t}")



for step in range(num_steps):
    t += dt
    dx = ufl.dx(domain)
    phi_avg = fem.assemble_scalar(fem.form(phi_n * dx)) / domain.comm.allreduce(domain.geometry.volume, op=MPI.SUM)

    # Вычитаем среднее значение из phi_n
    phi_n.x.array[:] -= phi_avg
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    solver_phi.solve(b, phi_n.x.petsc_vec)
    phi_n.x.scatter_forward()
    print(f"phi {phi_n.x.array[:]}, {t}")


    a2 = -ufl.dot(ufl.grad(p), ufl.grad(v1)) * ufl.dx
    L2 = phi_n * v1 * ufl.dx
    problem = LinearProblem(a2, L2, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    p_n = problem.solve()
    print(f"p {p_n.x.array[:]}, {t}")


    u_function.interpolate(compute_velocity(p_n))
    xdmf.write_function(p_h, t)
    xdmf_velocity.write_function(u_function, t)

xdmf.close()
xdmf_velocity.close()
