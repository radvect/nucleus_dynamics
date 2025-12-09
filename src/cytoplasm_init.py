import ufl
from dolfinx import fem, mesh, io

def init_cyto_var(msh):
    Vs = fem.functionspace(msh, ("CG", 1))          # scalar H1
    Vv = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    V_u = Vv
    V_P = Vs
    u = fem.Function(V_u, name="u")
    P = fem.Function(V_P, name="P")
    v_u = ufl.TestFunction(V_u)
