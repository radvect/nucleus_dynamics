import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
import meshio

try:
    import gmsh  # type: ignore
except ImportError:
    print("This script requires gmsh to be installed")
    exit(0)

# Инициализация Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

# Создаем модель
model = gmsh.model()
model.add("circle")

# Создаем окружность и область внутри неё
circle = model.occ.addCircle(0, 0, 0, 1)
loop = model.occ.addCurveLoop([circle])
surface = model.occ.addPlaneSurface([loop])

# Синхронизация модели
model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

# Добавляем физические группы
model.add_physical_group(dim=1, tags=[circle], name="Boundary")
model.add_physical_group(dim=2, tags=[surface], name="Interior")

# Генерация сетки
model.mesh.generate(dim=2)

# Сохраняем сетку в формате MSH
gmsh.write("circle.msh")
gmsh.finalize()

# Читаем созданную сетку и конвертируем в формат XDMF
mesh1 = meshio.read("circle.msh")
triangle_mesh = meshio.Mesh(
    points=mesh1.points,
    cells={"triangle": mesh1.cells_dict["triangle"]}
)
meshio.write("circle.xdmf", triangle_mesh)

# Загружаем сетку в DOLFINx
with io.XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

# Функциональные пространства
V = fem.functionspace(domain, ("Lagrange", 1))
U = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim,)))  
print(U.ufl_element())


#V_grad = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim,)))
#grad_p = fem.Function(V_grad)
#sgrad_p.name = "grad_p"


def initial_condition(x, a=-1, b = 1):
    return np.exp(-a * (x[0]**2 + b* x[1]**2))

p_n = fem.Function(V)
p_n.interpolate(initial_condition)

def source_function(x):
    return 0*np.exp(np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))

f = fem.Function(V)
f.interpolate(source_function)

p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
u = ufl.TrialFunction(U)

dt = 1.0 / 50  # Временной шаг
k = 1
C = 1
C1 = 1

# Уравнение для p
a = C*p * v * ufl.dx + k*dt * ufl.dot(ufl.grad(p), ufl.grad(v)) * ufl.dx
L = (p_n - dt * f) * v * ufl.dx



bilinear_form = fem.form(a)
linear_form = fem.form(L)


A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)


# Создаем два отдельных решателя
solver_p = PETSc.KSP().create(domain.comm)
solver_p.setOperators(A)
solver_p.setType(PETSc.KSP.Type.PREONLY)
solver_p.getPC().setType(PETSc.PC.Type.LU)





xdmf = io.XDMFFile(domain.comm, "pressure map.xdmf", "w")
xdmf_vector = io.XDMFFile(domain.comm, "velocity map.xdmf", "w")

xdmf.write_mesh(domain)
xdmf_vector.write_mesh(domain)

ph = fem.Function(V)
ph.name = "ph"
ph.interpolate(initial_condition)
xdmf.write_function(ph, 0.0)

p_n.x.array[:] = ph.x.array

u_ufl = -ufl.as_vector([ufl.grad(p_n)[0], ufl.grad(p_n)[1]])
u_expr = fem.Expression(u_ufl, U.element.interpolation_points())
u_function =fem.Function(U) 
u_function.name = "u"
u_function.interpolate(u_expr)
    
xdmf_vector.write_function(u_function, 0.0)

t = 0.0
num_steps = 10
for step in range(num_steps):
    t += dt

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    solver_p.solve(b, ph.x.petsc_vec)
    ph.x.scatter_forward()
    p_n.x.array[:] = ph.x.array

    print(U.ufl_element())

    
    u_ufl = -ufl.as_vector([ufl.grad(p_n)[0], ufl.grad(p_n)[1]])
    u_expr = fem.Expression(u_ufl, U.element.interpolation_points())
    u_function =fem.Function(U) 
    u_function.name = "u"
    u_function.interpolate(u_expr)
    
    xdmf.write_function(ph, t)
    xdmf_vector.write_function(u_function, t)
xdmf.close()
